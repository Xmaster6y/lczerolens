"""All concepts related to move.
"""

import chess
import torch

from lczerolens.game.wrapper import ModelWrapper, PolicyFlow
from lczerolens.utils import move as move_utils
from lczerolens.xai.concept import MulticlassConcept


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

        filtered_policy = torch.full((1858,), 0.0)
        legal_moves = [
            move_utils.encode_move(move, (board.turn, not board.turn))
            for move in board.legal_moves
        ]
        filtered_policy[legal_moves] = policy[legal_moves]
        return filtered_policy.argmax().item()
