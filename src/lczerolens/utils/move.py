"""
Utils for the move module.
"""

from typing import Tuple

import chess

from .constants import INVERTED_POLICY_INDEX, POLICY_INDEX


def encode_move(
    move: chess.Move,
    us_them: Tuple[bool, bool],
) -> int:
    """
    Converts a chess.Move object to an index.
    """
    us, _ = us_them
    from_square = move.from_square
    to_square = move.to_square

    if us == chess.BLACK:
        from_square_row = from_square // 8
        from_square_col = from_square % 8
        from_square = 8 * (7 - from_square_row) + from_square_col
        to_square_row = to_square // 8
        to_square_col = to_square % 8
        to_square = 8 * (7 - to_square_row) + to_square_col
    us_uci_move = (
        chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]
    )
    if move.promotion is not None:
        if move.promotion == chess.BISHOP:
            us_uci_move += "b"
        elif move.promotion == chess.ROOK:
            us_uci_move += "r"
        elif move.promotion == chess.QUEEN:
            us_uci_move += "q"
        # Knight promotion is the default
    return INVERTED_POLICY_INDEX[us_uci_move]


def decode_move(
    index: int,
    us_them: Tuple[bool, bool],
    board: chess.Board = chess.Board(),
) -> chess.Move:
    """
    Converts an index to a chess.Move object.
    """
    us, _ = us_them
    us_uci_move = POLICY_INDEX[index]
    from_square = chess.SQUARE_NAMES.index(us_uci_move[:2])
    to_square = chess.SQUARE_NAMES.index(us_uci_move[2:4])
    if us == chess.BLACK:
        from_square_row = from_square // 8
        from_square_col = from_square % 8
        from_square = 8 * (7 - from_square_row) + from_square_col
        to_square_row = to_square // 8
        to_square_col = to_square % 8
        to_square = 8 * (7 - to_square_row) + to_square_col

    uci_move = chess.SQUARE_NAMES[from_square] + chess.SQUARE_NAMES[to_square]
    from_piece = board.piece_at(from_square)
    if (
        from_piece == chess.PAWN and to_square >= 56
    ):  # Knight promotion is the default
        uci_move += "n"
    return chess.Move.from_uci(uci_move)
