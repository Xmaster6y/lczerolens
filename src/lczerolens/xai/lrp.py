"""
Compute LRP heatmap for a given model and input.
"""

import chess
import torch
import zennit

from lczerolens import board_utils
from lczerolens.adapt import ModelWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_lrp_heatmap(model, board: chess.Board):
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


class LrpWrapper(ModelWrapper):
    """
    Class for wrapping the LCZero models.
    """

    def compute_lrp_heatmap(self, board: chess.Board):
        """
        Runs basic LRP on the model.
        """
        relevance = compute_lrp_heatmap(self.model, board)
        return relevance
