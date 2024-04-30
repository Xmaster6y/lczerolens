"""
Gradio interface for plotting policy.
"""

import copy

import chess
import gradio as gr

from demo import constants, utils, visualisation

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
    plane_index,
    history_index,
):
    global cache
    global boards
    if model_name == "":
        gr.Warning("No model selected.")
        return None, None, None, None, None
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
    wrapper, lens = utils.get_wrapper_lens_from_state(model_name, "crp")
    cache = []
    for board in boards:
        relevance = lens.compute_heatmap(board, wrapper)
        cache.append(copy.deepcopy(relevance))
    return (
        *make_plot(
            plane_index,
        ),
        *make_history_plot(
            history_index,
        ),
    )


def make_plot(
    plane_index,
):
    global cache
    global boards
    global board_index

    if cache is None:
        gr.Warning("Cache not computed!")
        return None, None, None

    board = boards[board_index]
    relevance_tensor = cache[board_index]
    a_max = relevance_tensor.abs().max()
    if a_max != 0:
        relevance_tensor = relevance_tensor / a_max
    vmin = -1
    vmax = 1
    heatmap = relevance_tensor[plane_index - 1].view(64)
    if board.turn == chess.BLACK:
        heatmap = heatmap.view(8, 8).flip(0).view(64)
    svg_board, fig = visualisation.render_heatmap(board, heatmap, vmin=vmin, vmax=vmax)
    with open(f"{constants.FIGURE_DIRECTORY}/lrp.svg", "w") as f:
        f.write(svg_board)
    return f"{constants.FIGURE_DIRECTORY}/lrp.svg", board.fen(), fig


def make_history_plot(
    history_index,
):
    global cache
    global boards
    global board_index

    if cache is None:
        gr.Warning("Cache not computed!")
        return None, None

    board = boards[board_index]
    relevance_tensor = cache[board_index]
    a_max = relevance_tensor.abs().max()
    if a_max != 0:
        relevance_tensor = relevance_tensor / a_max
    vmin = -1
    vmax = 1
    heatmap = relevance_tensor[13 * (history_index - 1) : 13 * history_index - 1].sum(dim=0).view(64)
    if board.turn == chess.BLACK:
        heatmap = heatmap.view(8, 8).flip(0).view(64)
    if board_index - history_index + 1 < 0:
        history_board = chess.Board(fen=None)
    else:
        history_board = boards[board_index - history_index + 1]
    svg_board, fig = visualisation.render_heatmap(history_board, heatmap, vmin=vmin, vmax=vmax)
    with open(f"{constants.FIGURE_DIRECTORY}/lrp_history.svg", "w") as f:
        f.write(svg_board)
    return f"{constants.FIGURE_DIRECTORY}/lrp_history.svg", fig


def previous_board(
    plane_index,
    history_index,
):
    global board_index
    board_index -= 1
    if board_index < 0:
        gr.Warning("Already at first board.")
        board_index = 0
    return (
        *make_plot(
            plane_index,
        ),
        *make_history_plot(
            history_index,
        ),
    )


def next_board(
    plane_index,
    history_index,
):
    global board_index
    board_index += 1
    if board_index >= len(boards):
        gr.Warning("Already at last board.")
        board_index = len(boards) - 1
    return (
        *make_plot(
            plane_index,
        ),
        *make_history_plot(
            history_index,
        ),
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
                model_name = gr.Textbox(label="Selected model", lines=1, interactive=False, scale=7)

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
                value=("e2e3 b8c6 d2d4 e7e5 g1f3 d8e7 " "d4d5 e5e4 f3d4 c6e5 f2f4 e5g6"),
            )
            compute_cache_button = gr.Button("Compute heatmaps")

            with gr.Group():
                with gr.Row():
                    plane_index = gr.Slider(
                        label="Plane index",
                        minimum=1,
                        maximum=112,
                        step=1,
                        value=1,
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

    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    histroy_index = gr.Slider(
                        label="History index",
                        minimum=1,
                        maximum=8,
                        step=1,
                        value=1,
                    )
            history_colorbar = gr.Plot(label="Colorbar")
        with gr.Column():
            history_image = gr.Image(label="Board")

    base_inputs = [
        plane_index,
        histroy_index,
    ]
    outputs = [
        image,
        current_board_fen,
        colorbar,
        history_image,
        history_colorbar,
    ]

    compute_cache_button.click(
        compute_cache,
        inputs=[board_fen, action_seq, model_name] + base_inputs,
        outputs=outputs,
    )

    previous_board_button.click(previous_board, inputs=base_inputs, outputs=outputs)
    next_board_button.click(next_board, inputs=base_inputs, outputs=outputs)

    plane_index.change(
        make_plot,
        inputs=plane_index,
        outputs=[image, current_board_fen, colorbar],
    )
    histroy_index.change(
        make_history_plot,
        inputs=histroy_index,
        outputs=[history_image, history_colorbar],
    )
