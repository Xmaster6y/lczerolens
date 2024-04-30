"""All concepts related to move."""

import chess
import torch

from lczerolens.encodings import move as move_encodings
from lczerolens.model.wrapper import ModelWrapper, PolicyFlow
from lczerolens.xai.concept import BinaryConcept, MulticlassConcept


class BestLegalMoveConcept(MulticlassConcept):
    """Class for move concept-based XAI methods."""

    def __init__(
        self,
        wrapper: ModelWrapper,
    ):
        """Initialize the class."""
        self.policy_flow = PolicyFlow(wrapper.model)

    def compute_label(
        self,
        board: chess.Board,
    ) -> int:
        """Compute the label for a given model and input."""
        (policy,) = self.policy_flow.predict(board)
        policy = torch.softmax(policy.squeeze(0), dim=-1)

        legal_move_indices = [
            move_encodings.encode_move(move, (board.turn, not board.turn)) for move in board.legal_moves
        ]
        sub_index = policy[legal_move_indices].argmax().item()
        return legal_move_indices[sub_index]


class PieceBestLegalMoveConcept(BinaryConcept):
    """Class for move concept-based XAI methods."""

    def __init__(
        self,
        wrapper: ModelWrapper,
        piece: str,
    ):
        """Initialize the class."""
        self.policy_flow = PolicyFlow(wrapper.model)
        self.piece = chess.Piece.from_symbol(piece)

    def compute_label(
        self,
        board: chess.Board,
    ) -> int:
        """Compute the label for a given model and input."""
        (policy,) = self.policy_flow.predict(board)
        policy = torch.softmax(policy.squeeze(0), dim=-1)

        legal_moves = list(board.legal_moves)
        legal_move_indices = [move_encodings.encode_move(move, (board.turn, not board.turn)) for move in legal_moves]
        sub_index = policy[legal_move_indices].argmax().item()
        best_legal_move = legal_moves[sub_index]
        if board.piece_at(best_legal_move.from_square) == self.piece:
            return 1
        return 0
