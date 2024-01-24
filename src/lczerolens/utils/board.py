"""
Board utilities.
"""

import re
from copy import deepcopy
from typing import Optional, Tuple

import chess
import torch


def board_to_tensor13x8x8(
    board: chess.Board,
    us_them: Optional[Tuple[bool, bool]] = None,
):
    """
    Converts a chess.Board object to a 64 tensor.
    """
    if us_them is None:
        us = board.turn
        them = not us
    else:
        us, them = us_them
    plane_orders = {chess.WHITE: "PNBRQK", chess.BLACK: "pnbrqk"}
    plane_order = plane_orders[us] + plane_orders[them]

    def piece_to_index(piece: str):
        return f"{plane_order}0".index(piece)

    fen_board = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_board)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)

    tensor13x8x8 = torch.zeros((13, 8, 8), dtype=torch.float)
    ordinal_board = torch.tensor(
        tuple(map(piece_to_index, ordered_fen)), dtype=torch.float
    )
    ordinal_board = ordinal_board.reshape((8, 8)).unsqueeze(0)
    piece_tensor = torch.tensor(
        tuple(map(piece_to_index, plane_order)), dtype=torch.float
    )
    piece_tensor = piece_tensor.reshape((12, 1, 1))
    tensor13x8x8[:12] = (ordinal_board == piece_tensor).float()
    if board.is_repetition(2):
        tensor13x8x8[12] = torch.ones((8, 8), dtype=torch.float)
    return tensor13x8x8 if us == chess.WHITE else tensor13x8x8.flip(1)


def board_to_tensor112x8x8(
    last_board=chess.Board,
    with_history: bool = True,
):
    """
    Create the lc0 112x8x8 tensor from the history of a game.
    """
    board = deepcopy(last_board)
    tensor112x8x8 = torch.zeros((112, 8, 8), dtype=torch.float)
    us = last_board.turn
    them = not us
    if with_history:
        for i in range(8):
            tensor13x8x8 = board_to_tensor13x8x8(board, (us, them))
            tensor112x8x8[i * 13 : (i + 1) * 13] = tensor13x8x8
            try:
                board.pop()
            except IndexError:
                break
    if last_board.has_queenside_castling_rights(us):
        tensor112x8x8[104] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_kingside_castling_rights(us):
        tensor112x8x8[105] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_queenside_castling_rights(them):
        tensor112x8x8[106] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_kingside_castling_rights(them):
        tensor112x8x8[107] = torch.ones((8, 8), dtype=torch.float)
    if us == chess.BLACK:
        tensor112x8x8[108] = torch.ones((8, 8), dtype=torch.float)
    tensor112x8x8[109] = (
        torch.ones((8, 8), dtype=torch.float) * last_board.halfmove_clock
    )
    tensor112x8x8[111] = torch.ones((8, 8), dtype=torch.float)
    return tensor112x8x8
