"""
Gradio interface for visualizing the policy of a model.
"""

import chess
import chess.svg
import gradio as gr
import torch

from demo import constants, utils, visualisation
from lczerolens import move_utils
from lczerolens.xai import PolicyLens

current_board = None
current_raw_policy = None
current_policy = None
current_value = None
current_outcome = None


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


def compute_policy(
    board_fen,
    action_seq,
    model_name,
):
    global current_board
    global current_policy
    global current_raw_policy
    global current_value
    global current_outcome
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
        gr.Warning("Invalid FEN.")
        return (None, None, "", None)
    if action_seq:
        try:
            for action in action_seq.split():
                board.push_uci(action)
        except ValueError:
            gr.Warning("Invalid action sequence.")
            return (None, None, "", None)
    wrapper = utils.get_wrapper_from_state(model_name)
    (output,) = wrapper.predict(board)
    current_raw_policy = output["policy"][0]
    policy = torch.softmax(output["policy"][0], dim=-1)

    filtered_policy = torch.full((1858,), 0.0)
    legal_moves = [
        move_utils.encode_move(move, (board.turn, not board.turn))
        for move in board.legal_moves
    ]
    filtered_policy[legal_moves] = policy[legal_moves]
    policy = filtered_policy

    current_board = board
    current_policy = policy
    current_value = output.get("value", None)
    current_outcome = output.get("wdl", None)


def make_plot(
    view,
    aggregate_topk,
    move_to_play,
):
    global current_board
    global current_policy
    global current_raw_policy
    global current_value
    global current_outcome

    if current_board is None or current_policy is None:
        gr.Warning("Please compute a policy first.")
        return (None, None, "", None)

    pickup_agg, dropoff_agg = PolicyLens.aggregate_policy(
        current_policy, int(aggregate_topk)
    )

    if view == "from":
        if current_board.turn == chess.WHITE:
            heatmap = pickup_agg
        else:
            heatmap = pickup_agg.view(8, 8).flip(0).view(64)
    else:
        if current_board.turn == chess.WHITE:
            heatmap = dropoff_agg
        else:
            heatmap = dropoff_agg.view(8, 8).flip(0).view(64)
    us_them = (current_board.turn, not current_board.turn)
    topk_moves = torch.topk(current_policy, 50)
    move = move_utils.decode_move(
        topk_moves.indices[move_to_play - 1], us_them
    )
    arrows = [(move.from_square, move.to_square)]
    svg_board, fig = visualisation.render_heatmap(
        current_board, heatmap, arrows=arrows
    )
    with open(f"{constants.FIGURE_DIRECTORY}/policy.svg", "w") as f:
        f.write(svg_board)
    fig_dist = visualisation.render_policy_distribution(
        current_raw_policy,
        [
            move_utils.encode_move(move, us_them)
            for move in current_board.legal_moves
        ],
    )
    return (
        f"{constants.FIGURE_DIRECTORY}/policy.svg",
        fig,
        (f"Value: {current_value} - WDL: {current_outcome}"),
        fig_dist,
    )


def make_policy_plot(
    board_fen,
    action_seq,
    view,
    model_name,
    aggregate_topk,
    move_to_play,
):
    compute_policy(
        board_fen,
        action_seq,
        model_name,
    )
    return make_plot(
        view,
        aggregate_topk,
        move_to_play,
    )


def play_move(
    board_fen,
    action_seq,
    view,
    model_name,
    aggregate_topk,
    move_to_play,
):
    global current_board
    global current_policy

    move = move_utils.decode_move(
        current_policy.topk(50).indices[move_to_play - 1],
        (current_board.turn, not current_board.turn),
    )
    current_board.push(move)
    action_seq = f"{action_seq} {move.uci()}"
    compute_policy(
        board_fen,
        action_seq,
        model_name,
    )
    return [
        *make_plot(
            view,
            aggregate_topk,
            1,
        ),
        action_seq,
        1,
    ]


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
                value=(
                    "e2e3 b8c6 d2d4 e7e5 g1f3 d8e7 "
                    "d4d5 e5e4 f3d4 c6e5 f2f4 e5g6"
                ),
            )
            with gr.Group():
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
                    move_to_play = gr.Slider(
                        label="Move to play",
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=1,
                        scale=3,
                    )
                    play_button = gr.Button("Play")

            policy_button = gr.Button("Compute policy")
            colorbar = gr.Plot(label="Colorbar")
            game_info = gr.Textbox(
                label="Game info", lines=1, max_lines=1, value=""
            )
        with gr.Column():
            image = gr.Image(label="Board")
            density_plot = gr.Plot(label="Density")

    policy_inputs = [
        board_fen,
        action_seq,
        view,
        model_name,
        aggregate_topk,
        move_to_play,
    ]
    policy_outputs = [image, colorbar, game_info, density_plot]
    policy_button.click(
        make_policy_plot, inputs=policy_inputs, outputs=policy_outputs
    )
    board_fen.submit(
        make_policy_plot, inputs=policy_inputs, outputs=policy_outputs
    )
    action_seq.submit(
        make_policy_plot, inputs=policy_inputs, outputs=policy_outputs
    )

    fast_inputs = [
        view,
        aggregate_topk,
        move_to_play,
    ]
    aggregate_topk.change(
        make_plot, inputs=fast_inputs, outputs=policy_outputs
    )
    view.change(make_plot, inputs=fast_inputs, outputs=policy_outputs)
    move_to_play.change(make_plot, inputs=fast_inputs, outputs=policy_outputs)

    play_button.click(
        play_move,
        inputs=policy_inputs,
        outputs=policy_outputs + [action_seq, move_to_play],
    )
