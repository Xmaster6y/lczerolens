"""Board utilities."""

import re
from copy import deepcopy
from enum import Enum
from typing import Optional

import chess
import torch


class InputEncoding(int, Enum):
    """Input encoding for the board tensor."""

    INPUT_CLASSICAL_112_PLANE = 0
    INPUT_CLASSICAL_112_PLANE_REPEATED = 1
    INPUT_CLASSICAL_112_PLANE_NO_HISTORY = 2


def get_plane_order(us: bool):
    """Get the plane order for the given us view.

    Parameters
    ----------
    us : bool
        The us_them tuple.

    Returns
    -------
    str
        The plane order.
    """
    plane_orders = {chess.WHITE: "PNBRQK", chess.BLACK: "pnbrqk"}
    plane_order = plane_orders[us] + plane_orders[not us]
    return plane_order


def get_piece_index(piece: str, us: bool, plane_order: Optional[str] = None):
    """Converts a piece to its index in the plane order.

    Parameters
    ----------
    piece : str
        The piece to convert.
    us : bool
        The us_them tuple.
    plane_order : Optional[str]
        The plane order.

    Returns
    -------
    int
        The index of the piece in the plane order.
    """
    if plane_order is None:
        plane_order = get_plane_order(us)
    return f"{plane_order}0".index(piece)


def board_to_config_tensor(
    board: chess.Board,
    us: Optional[bool] = None,
    input_encoding: InputEncoding = InputEncoding.INPUT_CLASSICAL_112_PLANE,
):
    """Converts a chess.Board to a tensor based on the pieces configuration.

    Parameters
    ----------
    board : chess.Board
        The board to convert.
    us : Optional[bool]
        The us_them tuple.
    input_encoding : InputEncoding
        The input encoding method.

    Returns
    -------
    torch.Tensor
        The 13x8x8 tensor.
    """
    if not isinstance(input_encoding, InputEncoding):
        raise NotImplementedError(f"Input encoding {input_encoding} not implemented.")
    if us is None:
        us = board.turn
    plane_order = get_plane_order(us)

    def piece_to_index(piece: str):
        return f"{plane_order}0".index(piece)

    fen_board = board.fen().split(" ")[0]
    fen_rep = re.sub(r"(\d)", lambda m: "0" * int(m.group(1)), fen_board)
    rows = fen_rep.split("/")
    rev_rows = rows[::-1]
    ordered_fen = "".join(rev_rows)

    config_tensor = torch.zeros((13, 8, 8), dtype=torch.float)
    ordinal_board = torch.tensor(tuple(map(piece_to_index, ordered_fen)), dtype=torch.float)
    ordinal_board = ordinal_board.reshape((8, 8)).unsqueeze(0)
    piece_tensor = torch.tensor(tuple(map(piece_to_index, plane_order)), dtype=torch.float)
    piece_tensor = piece_tensor.reshape((12, 1, 1))
    config_tensor[:12] = (ordinal_board == piece_tensor).float()
    if board.is_repetition(2):  # Might be wrong if the full history is not available
        config_tensor[12] = torch.ones((8, 8), dtype=torch.float)
    return config_tensor if us == chess.WHITE else config_tensor.flip(1)


def board_to_input_tensor(
    last_board=chess.Board,
    with_history: bool = True,
    input_encoding: InputEncoding = InputEncoding.INPUT_CLASSICAL_112_PLANE,
):
    """Create the lc0 input tensor from the history of a game.

    Parameters
    ----------
    last_board : chess.Board
        The last board in the game.
    with_history : bool
        Whether to include the history of the game.
    input_encoding : InputEncoding
        The input encoding method.

    Returns
    -------
    torch.Tensor
        The 112x8x8 tensor.
    """
    if not isinstance(input_encoding, InputEncoding):
        raise NotImplementedError(f"Input encoding {input_encoding} not implemented.")
    board = deepcopy(last_board)
    input_tensor = torch.zeros((112, 8, 8), dtype=torch.float)
    us = last_board.turn
    them = not us
    if with_history:
        if (
            input_encoding == InputEncoding.INPUT_CLASSICAL_112_PLANE
            or input_encoding == InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED
        ):
            for i in range(8):
                config_tensor = board_to_config_tensor(board, us)
                input_tensor[i * 13 : (i + 1) * 13] = config_tensor
                try:
                    board.pop()
                except IndexError:
                    if input_encoding == InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED:
                        input_tensor[(i + 1) * 13 : 104] = config_tensor.repeat(7 - i, 1, 1)
                    break

        elif input_encoding == InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY:
            config_tensor = board_to_config_tensor(board, us)
            input_tensor[:104] = config_tensor.repeat(8, 1, 1)
        else:
            raise ValueError(f"Got unexpected input encoding {input_encoding}")
    if last_board.has_queenside_castling_rights(us):
        input_tensor[104] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_kingside_castling_rights(us):
        input_tensor[105] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_queenside_castling_rights(them):
        input_tensor[106] = torch.ones((8, 8), dtype=torch.float)
    if last_board.has_kingside_castling_rights(them):
        input_tensor[107] = torch.ones((8, 8), dtype=torch.float)
    if us == chess.BLACK:
        input_tensor[108] = torch.ones((8, 8), dtype=torch.float)
    input_tensor[109] = torch.ones((8, 8), dtype=torch.float) * last_board.halfmove_clock
    input_tensor[111] = torch.ones((8, 8), dtype=torch.float)
    return input_tensor
