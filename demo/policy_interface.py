"""
Gradio interface for visualizing the policy of a model.
"""

import chess
import chess.svg
import gradio as gr
import torch

from demo import constants, utils
from lczerolens import model_utils, move_utils, visualisation_utils


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


def make_policy_plot(
    board_fen,
    action_seq,
    view,
    model_name,
    depth,
    use_softmax,
    aggregate_topk,
    render_bestk,
    only_legal,
):
    if model_name == "":
        gr.Warning(
            "Please select a model.",
        )
        return (
            None,
            None,
            "",
        )
    try:
        board = chess.Board(board_fen)
    except ValueError:
        board = chess.Board()
        gr.Warning("Invalid FEN, using starting position.")
    if action_seq:
        try:
            for action in action_seq.split():
                board.push_uci(action)
        except ValueError:
            gr.Warning("Invalid action sequence, using starting position.")
            board = chess.Board()
    wrapper = utils.get_wrapper_from_state(model_name)
    policy, outcome, value = wrapper.prediction(board)
    if use_softmax:
        policy = torch.softmax(policy, dim=-1)
    value = value.item()
    us_win = outcome[0].item()
    draw = outcome[1].item()
    them_win = outcome[2].item()
    pickup_agg, dropoff_agg = model_utils.aggregate_policy(
        policy, int(aggregate_topk)
    )

    if view == "from":
        if board.turn == chess.WHITE:
            heatmap = pickup_agg
        else:
            heatmap = pickup_agg.view(8, 8).flip(0).view(64)
    else:
        if board.turn == chess.WHITE:
            heatmap = dropoff_agg
        else:
            heatmap = dropoff_agg.view(8, 8).flip(0).view(64)
    us_them = (board.turn, not board.turn)
    if only_legal:
        legal_moves = [
            move_utils.encode_move(move, us_them) for move in board.legal_moves
        ]
        filtered_policy = torch.zeros(1858)
        filtered_policy[legal_moves] = policy[legal_moves]
        if (filtered_policy < 0).any():
            gr.Warning("Some legal moves have negative policy.")
        topk_moves = torch.topk(filtered_policy, render_bestk)
    else:
        topk_moves = torch.topk(policy, render_bestk)
    arrows = []
    for move_index in topk_moves.indices:
        move = move_utils.decode_move(move_index, us_them)
        arrows.append((move.from_square, move.to_square))
    svg_board, fig = visualisation_utils.render_heatmap(
        board, heatmap, arrows=arrows
    )
    with open(f"{constants.FIGURE_DIRECTORY}/policy.svg", "w") as f:
        f.write(svg_board)
    return (
        f"{constants.FIGURE_DIRECTORY}/policy.svg",
        fig,
        (
            f"Value: {value:.2f} - Win: {us_win:.2f} - "
            f"Draw: {draw:.2f} - Loss: {them_win:.2f}"
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
                label="Board FEN",
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
            with gr.Group():
                with gr.Row():
                    depth = gr.Radio(label="Depth", choices=[0], value=0)
                    use_softmax = gr.Checkbox(label="Use softmax", value=True)
                with gr.Row():
                    aggregate_topk = gr.Slider(
                        label="Aggregate top k",
                        minimum=1,
                        maximum=1858,
                        step=1,
                        value=1858,
                        scale=3,
                    )
                    view = gr.Radio(
                        label="View",
                        choices=["from", "to"],
                        value="from",
                        scale=1,
                    )
                with gr.Row():
                    render_bestk = gr.Slider(
                        label="Render best k",
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=5,
                        scale=3,
                    )
                    only_legal = gr.Checkbox(
                        label="Only legal", value=True, scale=1
                    )

            policy_button = gr.Button("Plot policy")
            colorbar = gr.Plot(label="Colorbar")
            game_info = gr.Textbox(
                label="Game info", lines=1, max_lines=1, value=""
            )
        with gr.Column():
            image = gr.Image(label="Board")

    policy_inputs = [
        board_fen,
        action_seq,
        view,
        model_name,
        depth,
        use_softmax,
        aggregate_topk,
        render_bestk,
        only_legal,
    ]
    policy_outputs = [image, colorbar, game_info]
    policy_button.click(
        make_policy_plot, inputs=policy_inputs, outputs=policy_outputs
    )
