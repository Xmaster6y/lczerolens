"""
Compute LRP heatmap for a given model and input.
"""

from typing import Any, Dict

import chess
import torch
import zennit

from lczerolens import board_utils
from lczerolens.adapt import ModelWrapper

from .lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LrpLens(Lens):
    """
    Class for wrapping the LCZero models.
    """

    def compute_heatmap(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Runs basic LRP on the model.
        """
        relevance = self._compute_lrp_heatmap(wrapper.model, board)
        return relevance

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        return True

    def _compute_lrp_heatmap(self, model, board: chess.Board):
        """
        Compute LRP heatmap for a given model and input.
        """

        board_tensor = board_utils.board_to_tensor112x8x8(board)
        board_tensor.to(DEVICE)
        model.to(DEVICE)
        model.eval()
        board_tensor.requires_grad_(True)

        composite = zennit.composites.EpsilonPlusFlat()
        with composite.context(model) as modified_model:
            out = modified_model(input)
            if len(out) == 2:
                policy, outcome_probs = out
                value = torch.zeros(outcome_probs.shape[0], 1)
            else:
                policy, outcome_probs, value = out
            # gradient/ relevance wrt. class/output 0
            outcome_probs.backward(gradient=torch.eye(3)[[0]])
        relevance = board_tensor.grad[0]
        return relevance
