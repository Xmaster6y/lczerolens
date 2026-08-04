"""Stateless Lczero input-plane encoding."""

from __future__ import annotations

import re
from enum import Enum

import chess
import torch


class InputFormat(str, Enum):
    """Supported 112-plane history policies."""

    CLASSICAL_112 = "classical_112"
    CLASSICAL_112_REPEATED = "classical_112_repeated"
    NO_HISTORY_REPEATED = "no_history_repeated"
    NO_HISTORY_ZEROS = "no_history_zeros"


def encode_input(
    board: chess.Board,
    *,
    input_format: InputFormat = InputFormat.CLASSICAL_112,
) -> torch.Tensor:
    """Encode ``board`` as one Lczero ``[112, 8, 8]`` input tensor.

    The caller's board and move stack are never mutated.  History-sensitive
    formats operate on a private full-stack copy.
    """
    _validate_board(board)
    if not isinstance(input_format, InputFormat):
        raise TypeError("input_format must be an InputFormat.")

    encoded = torch.zeros((112, 8, 8), dtype=torch.float32)
    us = board.turn
    them = not us

    if input_format in {InputFormat.CLASSICAL_112, InputFormat.CLASSICAL_112_REPEATED}:
        working = board.copy(stack=True)
        for index in range(8):
            configuration = _encode_configuration(working, us)
            encoded[index * 13 : (index + 1) * 13] = configuration
            if not working.move_stack:
                if input_format is InputFormat.CLASSICAL_112_REPEATED:
                    encoded[(index + 1) * 13 : 104] = configuration.repeat(7 - index, 1, 1)
                break
            working.pop()
    elif input_format is InputFormat.NO_HISTORY_REPEATED:
        encoded[:104] = _encode_configuration(board, us).repeat(8, 1, 1)
    elif input_format is InputFormat.NO_HISTORY_ZEROS:
        encoded[:13] = _encode_configuration(board, us)

    if board.has_queenside_castling_rights(us):
        encoded[104].fill_(1)
    if board.has_kingside_castling_rights(us):
        encoded[105].fill_(1)
    if board.has_queenside_castling_rights(them):
        encoded[106].fill_(1)
    if board.has_kingside_castling_rights(them):
        encoded[107].fill_(1)
    if us == chess.BLACK:
        encoded[108].fill_(1)
    encoded[109].fill_(board.halfmove_clock)
    encoded[111].fill_(1)
    return encoded


def _encode_configuration(board: chess.Board, perspective: chess.Color) -> torch.Tensor:
    plane_order = _plane_order(perspective)
    board_fen = board.board_fen()
    expanded = re.sub(r"(\d)", lambda match: "0" * int(match.group(1)), board_fen)
    ordered = "".join(expanded.split("/")[::-1])

    ordinal_board = torch.tensor(tuple(f"{plane_order}0".index(piece) for piece in ordered)).reshape(1, 8, 8)
    piece_indices = torch.tensor(tuple(f"{plane_order}0".index(piece) for piece in plane_order)).reshape(12, 1, 1)
    configuration = torch.zeros((13, 8, 8), dtype=torch.float32)
    configuration[:12] = (ordinal_board == piece_indices).to(torch.float32)
    if board.is_repetition(2):
        configuration[12].fill_(1)
    return configuration if perspective == chess.WHITE else configuration.flip(1)


def _plane_order(perspective: chess.Color) -> str:
    first = "PNBRQK" if perspective == chess.WHITE else "pnbrqk"
    second = "pnbrqk" if perspective == chess.WHITE else "PNBRQK"
    return first + second


def _validate_board(board: chess.Board) -> None:
    if not isinstance(board, chess.Board):
        raise TypeError(f"Expected chess.Board, got {type(board).__name__}.")


__all__ = ["InputFormat", "encode_input"]
