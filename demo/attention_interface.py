"""
Gradio interface for plotting attention.
"""

import copy

import chess
import gradio as gr

from demo import constants, utils
from lczerolens import visualisation_utils

cache = None
boards = None
board_index = 0


def list_models():
    """
    List the models in the model directory.
    """
    models_info = utils.get_models_info(leela=False)
    return sorted([[model_info[0]] for model_info in models_info])


def on_select_model_df(
    evt: gr.SelectData,
):
    """
    When a model is selected, update the statement.
    """
    return evt.value


def compute_cache(
    board_fen,
    action_seq,
    model_name,
    attention_layer,
    attention_head,
    square,
    quantity,
    func,
    trick,
    aggregate,
):
    global cache
    global boards
    if model_name == "":
        gr.Warning("No model selected.")
        return None, None, None

    try:
        board = chess.Board(board_fen)
    except ValueError:
        board = chess.Board()
        gr.Warning("Invalid FEN, using starting position.")
    boards = [board.copy()]
    if action_seq:
        try:
            if action_seq.startswith("1."):
                for action in action_seq.split():
                    if action.endswith("."):
                        continue
                    board.push_san(action)
                    boards.append(board.copy())
            else:
                for action in action_seq.split():
                    board.push_uci(action)
                    boards.append(board.copy())
        except ValueError:
            gr.Warning(f"Invalid action {action} stopping before it.")
    try:
        wrapper, lens = utils.get_wrapper_lens_from_state(
            model_name, "attention"
        )
    except ValueError:
        gr.Warning("Could not load model.")
        return None, None, None
    cache = []
    for board in boards:
        attention_cache = copy.deepcopy(lens.compute_heatmap(board, wrapper))
        cache.append(attention_cache)
    return make_plot(
        attention_layer,
        attention_head,
        square,
        quantity,
        func,
        trick,
        aggregate,
    )


def make_plot(
    attention_layer, attention_head, square, quantity, func, trick, aggregate
):
    global cache
    global boards
    global board_index

    if cache is None:
        gr.Warning("Cache not computed!")
        return None, None

    board = boards[board_index]
    num_attention_layers = len(cache[board_index])
    if attention_layer > num_attention_layers:
        gr.Warning(
            f"Attention layer {attention_layer} does not exist, "
            f"using layer {num_attention_layers} instead."
        )
        attention_layer = num_attention_layers

    key = f"{attention_layer-1}-{quantity}-{func}"
    try:
        attention_tensor = cache[board_index][key]
    except KeyError:
        gr.Warning(f"Combination {key} does not exist.")
        return None, None, None
    if attention_head > attention_tensor.shape[1]:
        gr.Warning(
            f"Attention head {attention_head} does not exist, "
            f"using head {attention_tensor.shape[1]} instead."
        )
        attention_head = attention_tensor.shape[1]
    try:
        square_index = chess.SQUARE_NAMES.index(square)
    except ValueError:
        gr.Warning(f"Invalid square {square}, using a1 instead.")
        square_index = 0
        square = "a1"
    if board.turn == chess.BLACK:
        square_index = chess.square_mirror(square_index)

    if trick == "revert":
        square_index = 63 - square_index

    if aggregate == "Row":
        heatmap = attention_tensor[0, attention_head - 1, square_index, :]
    elif aggregate == "Column":
        heatmap = attention_tensor[0, attention_head - 1, :, square_index]
    else:
        heatmap = attention_tensor[0, attention_head - 1]
    if board.turn == chess.BLACK:
        heatmap = heatmap.view(8, 8).flip(0).view(64)
    svg_board, fig = visualisation_utils.render_heatmap(
        board, heatmap, square=square
    )
    with open(f"{constants.FIGURE_DIRECTORY}/attention.svg", "w") as f:
        f.write(svg_board)
    return f"{constants.FIGURE_DIRECTORY}/attention.svg", board.fen(), fig


def previous_board(
    attention_layer,
    attention_head,
    square,
    from_to,
    color_flip,
    trick,
    aggregate,
):
    global board_index
    board_index -= 1
    if board_index < 0:
        gr.Warning("Already at first board.")
        board_index = 0
    return make_plot(
        attention_layer, attention_head, square, from_to, color_flip
    )


def next_board(
    attention_layer,
    attention_head,
    square,
    from_to,
    color_flip,
    trick,
    aggregate,
):
    global board_index
    board_index += 1
    if board_index >= len(boards):
        gr.Warning("Already at last board.")
        board_index = len(boards) - 1
    return make_plot(
        attention_layer, attention_head, square, from_to, color_flip
    )


with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column(scale=2):
            model_df = gr.Dataframe(
                headers=["Available models"],
                datatype=["str"],
                interactive=False,
                type="array",
                value=list_models,
            )
        with gr.Column(scale=1):
            with gr.Row():
                model_name = gr.Textbox(
                    label="Selected model", lines=1, interactive=False, scale=7
                )

    model_df.select(
        on_select_model_df,
        None,
        model_name,
    )

    with gr.Row():
        with gr.Column():
            board_fen = gr.Textbox(
                label="Board starting FEN",
                lines=1,
                max_lines=1,
                value=chess.STARTING_FEN,
            )
            action_seq = gr.Textbox(
                label="Action sequence",
                lines=1,
                max_lines=1,
                value=(
                    "e2e3 b8c6 d2d4 e7e5 g1f3 d8e7 "
                    "d4d5 e5e4 f3d4 c6e5 f2f4 e5g6"
                ),
            )
            compute_cache_button = gr.Button("Compute cache")

            with gr.Group():
                with gr.Row():
                    attention_layer = gr.Slider(
                        label="Attention layer",
                        minimum=1,
                        maximum=24,
                        step=1,
                        value=1,
                    )
                    attention_head = gr.Slider(
                        label="Attention head",
                        minimum=1,
                        maximum=24,
                        step=1,
                        value=1,
                    )
                with gr.Row():
                    square = gr.Textbox(
                        label="Square",
                        lines=1,
                        max_lines=1,
                        value="a1",
                        scale=1,
                    )
                    quantity = gr.Dropdown(
                        label="Quantity",
                        choices=["QK", "Q", "K", "out", "QKV"],
                        value="QK",
                        scale=2,
                    )
                    aggregate = gr.Dropdown(
                        label="Aggregate",
                        choices=["Row", "Column", "None"],
                        value="Row",
                        scale=2,
                    )
                    func = gr.Dropdown(
                        label="Function",
                        choices=[
                            "softmax",
                            "transpose",
                            "matmul",
                            "scale",
                        ],
                        value="softmax",
                        scale=2,
                    )
                    trick = gr.Dropdown(
                        label="Trick",
                        choices=[
                            "none",
                            "revert",
                        ],
                        value="none",
                        scale=2,
                    )
                with gr.Row():
                    previous_board_button = gr.Button("Previous board")
                    next_board_button = gr.Button("Next board")
            current_board_fen = gr.Textbox(
                label="Board FEN",
                lines=1,
                max_lines=1,
            )
            colorbar = gr.Plot(label="Colorbar")
        with gr.Column():
            image = gr.Image(label="Board")

    base_inputs = [
        attention_layer,
        attention_head,
        square,
        quantity,
        func,
        trick,
        aggregate,
    ]
    outputs = [image, current_board_fen, colorbar]

    compute_cache_button.click(
        compute_cache,
        inputs=[board_fen, action_seq, model_name] + base_inputs,
        outputs=outputs,
    )

    previous_board_button.click(
        previous_board, inputs=base_inputs, outputs=outputs
    )
    next_board_button.click(next_board, inputs=base_inputs, outputs=outputs)

    attention_layer.change(make_plot, inputs=base_inputs, outputs=outputs)
    attention_head.change(make_plot, inputs=base_inputs, outputs=outputs)
    square.submit(make_plot, inputs=base_inputs, outputs=outputs)
    quantity.change(make_plot, inputs=base_inputs, outputs=outputs)
    func.change(make_plot, inputs=base_inputs, outputs=outputs)
    trick.change(make_plot, inputs=base_inputs, outputs=outputs)
    aggregate.change(make_plot, inputs=base_inputs, outputs=outputs)
