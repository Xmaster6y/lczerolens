"""Stateless mapping between chess moves and the Lczero policy vocabulary."""

from __future__ import annotations

import chess
import torch

from lczerolens.constants import INVERTED_POLICY_INDEX, POLICY_INDEX


POLICY_SIZE = len(POLICY_INDEX)


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """Return the fixed Lczero policy index for ``move`` in ``board`` context."""
    _validate_board(board)
    if not isinstance(move, chess.Move):
        raise TypeError(f"Expected chess.Move, got {type(move).__name__}.")

    policy_move = _policy_move(board, move)
    from_square = _relative_square(policy_move.from_square, board.turn)
    to_square = _relative_square(policy_move.to_square, board.turn)
    label = chess.square_name(from_square) + chess.square_name(to_square)
    if move.promotion == chess.BISHOP:
        label += "b"
    elif move.promotion == chess.ROOK:
        label += "r"
    elif move.promotion == chess.QUEEN:
        label += "q"
    return INVERTED_POLICY_INDEX[label]


def decode_move(board: chess.Board, index: int) -> chess.Move:
    """Decode one policy index in ``board`` context."""
    _validate_board(board)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < POLICY_SIZE:
        raise ValueError(f"Policy index must be an integer in [0, {POLICY_SIZE}), got {index!r}.")

    label = POLICY_INDEX[index]
    from_square = _absolute_square(chess.parse_square(label[:2]), board.turn)
    to_square = _absolute_square(chess.parse_square(label[2:4]), board.turn)
    promotion = None
    if len(label) == 5:
        promotion = {"b": chess.BISHOP, "r": chess.ROOK, "q": chess.QUEEN}[label[4]]
    else:
        piece = board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to_square) in {0, 7}:
            promotion = chess.KNIGHT
    decoded = chess.Move(from_square, to_square, promotion=promotion)
    if not board.chess960:
        for legal_move in board.legal_moves:
            if board.is_castling(legal_move) and _policy_move(board, legal_move) == decoded:
                return legal_move
    return decoded


def legal_indices(board: chess.Board) -> torch.Tensor:
    """Return policy indices for legal moves in python-chess iteration order."""
    _validate_board(board)
    return torch.tensor([encode_move(board, move) for move in board.legal_moves], dtype=torch.long)


def legal_mask(board: chess.Board) -> torch.Tensor:
    """Return a fixed-size legal-action mask, empty for terminal positions."""
    _validate_board(board)
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)
    if board.is_game_over():
        return mask
    mask[legal_indices(board)] = True
    return mask


def _relative_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def _absolute_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def _policy_move(board: chess.Board, move: chess.Move) -> chess.Move:
    """Translate standard castling to Lczero's king-to-rook vocabulary."""
    if board.chess960 or not board.is_castling(move):
        return move
    rank = chess.square_rank(move.from_square)
    rook_file = 7 if board.is_kingside_castling(move) else 0
    return chess.Move(move.from_square, chess.square(rook_file, rank))


def _validate_board(board: chess.Board) -> None:
    if not isinstance(board, chess.Board):
        raise TypeError(f"Expected chess.Board, got {type(board).__name__}.")


__all__ = ["POLICY_SIZE", "decode_move", "encode_move", "legal_indices", "legal_mask"]
