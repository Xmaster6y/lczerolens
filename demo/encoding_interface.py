"""
Gradio interface for plotting encodings.
"""

import chess
import gradio as gr

from demo import constants
from lczerolens import board_utils, visualisation_utils


def make_encoding_plot(
    board_fen,
    action_seq,
    plane_index,
    color_flip,
):
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
    board_tensor = board_utils.board_to_tensor112x8x8(board)
    heatmap = board_tensor[plane_index]
    if color_flip and board.turn == chess.BLACK:
        heatmap = heatmap.flip(0)
    svg_board, fig = visualisation_utils.render_heatmap(
        board, heatmap.view(64), vmin=0.0, vmax=1.0
    )
    with open(f"{constants.FIGURE_DIRECTORY}/encoding.svg", "w") as f:
        f.write(svg_board)
    return f"{constants.FIGURE_DIRECTORY}/encoding.svg", fig


with gr.Blocks() as interface:
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
            with gr.Group():
                with gr.Row():
                    plane_index = gr.Slider(
                        label="Plane index",
                        minimum=0,
                        maximum=111,
                        step=1,
                        value=0,
                        scale=3,
                    )
                    color_flip = gr.Checkbox(
                        label="Color flip", value=True, scale=1
                    )

            colorbar = gr.Plot(label="Colorbar")
        with gr.Column():
            image = gr.Image(label="Board")

    policy_inputs = [
        board_fen,
        action_seq,
        plane_index,
        color_flip,
    ]
    policy_outputs = [image, colorbar]
    board_fen.submit(
        make_encoding_plot, inputs=policy_inputs, outputs=policy_outputs
    )
    action_seq.submit(
        make_encoding_plot, inputs=policy_inputs, outputs=policy_outputs
    )
    plane_index.change(
        make_encoding_plot, inputs=policy_inputs, outputs=policy_outputs
    )
    color_flip.change(
        make_encoding_plot, inputs=policy_inputs, outputs=policy_outputs
    )
    interface.load(
        make_encoding_plot, inputs=policy_inputs, outputs=policy_outputs
    )
