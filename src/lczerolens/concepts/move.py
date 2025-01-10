"""All concepts related to move."""

import chess
import torch

from lczerolens.board import LczeroBoard
from lczerolens.model import LczeroModel, PolicyFlow
from lczerolens.concept import BinaryConcept, MulticlassConcept


class BestLegalMove(MulticlassConcept):
    """Class for move concept-based XAI methods."""

    def __init__(
        self,
        model: LczeroModel,
    ):
        """Initialize the class."""
        self.policy_flow = PolicyFlow(model)

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """Compute the label for a given model and input."""
        (policy,) = self.policy_flow(board)
        policy = torch.softmax(policy.squeeze(0), dim=-1)

        legal_move_indices = [LczeroBoard.encode_move(move, board.turn) for move in board.legal_moves]
        sub_index = policy[legal_move_indices].argmax().item()
        return legal_move_indices[sub_index]


class PieceBestLegalMove(BinaryConcept):
    """Class for move concept-based XAI methods."""

    def __init__(
        self,
        model: LczeroModel,
        piece: str,
    ):
        """Initialize the class."""
        self.policy_flow = PolicyFlow(model)
        self.piece = chess.Piece.from_symbol(piece)

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """Compute the label for a given model and input."""
        (policy,) = self.policy_flow(board)
        policy = torch.softmax(policy.squeeze(0), dim=-1)

        legal_moves = list(board.legal_moves)
        legal_move_indices = [LczeroBoard.encode_move(move, board.turn) for move in legal_moves]
        sub_index = policy[legal_move_indices].argmax().item()
        best_legal_move = legal_moves[sub_index]
        if board.piece_at(best_legal_move.from_square) == self.piece:
            return 1
        return 0
