"""
Gradio interface for plotting a board.
"""

import chess
import gradio as gr

from demo import constants


def make_board_plot(
    board_fen,
    arrows,
):
    try:
        board = chess.Board(board_fen)
    except ValueError:
        board = chess.Board()
        gr.Warning("Invalid FEN, using starting position.")
    try:
        if arrows:
            arrows_list = arrows.split(" ")
            chess_arrows = []
            for arrow in arrows_list:
                from_square, to_square = arrow[:2], arrow[2:]
                chess_arrows.append(
                    (
                        chess.parse_square(from_square),
                        chess.parse_square(to_square),
                    )
                )
        else:
            chess_arrows = []
    except ValueError:
        chess_arrows = []
        gr.Warning("Invalid arrows, using none.")

    svg_board = chess.svg.board(
        board,
        size=350,
        arrows=chess_arrows,
    )
    with open(f"{constants.FIGURE_DIRECTORY}/board.svg", "w") as f:
        f.write(svg_board)
    return f"{constants.FIGURE_DIRECTORY}/board.svg"


with gr.Blocks() as interface:
    with gr.Row():
        with gr.Column():
            board_fen = gr.Textbox(
                label="Board starting FEN",
                lines=1,
                max_lines=1,
                value=chess.STARTING_FEN,
            )
            arrows = gr.Textbox(
                label="Arrows",
                lines=1,
                max_lines=1,
                value="",
                placeholder="e2e4 e7e5",
            )
        with gr.Column():
            image = gr.Image(label="Board", interactive=False)

    inputs = [
        board_fen,
        arrows,
    ]
    board_fen.submit(make_board_plot, inputs=inputs, outputs=image)
    arrows.submit(make_board_plot, inputs=inputs, outputs=image)
    interface.load(make_board_plot, inputs=inputs, outputs=image)
