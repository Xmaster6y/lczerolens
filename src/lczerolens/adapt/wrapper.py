"""
Class for wrapping the LCZero models.
"""

from typing import List

import chess
from torch import nn

from lczerolens import prediction_utils

from .builder import AutoBuilder


class ModelWrapper:
    """
    Class for wrapping the LCZero models.
    """

    model_builder = AutoBuilder

    def __init__(
        self,
        model: nn.Module,
    ):
        """
        Initializes the wrapper.
        """
        if not isinstance(model, nn.Module):
            raise ValueError(
                f"Model must be an instance of LczeroModel, not {type(model)}"
            )
        self.model = model

    @classmethod
    def from_path(cls, model_path):
        """
        Creates a wrapper from a model path.
        """
        model = cls.model_builder.build_from_path(model_path)
        return cls(model)

    def __call__(self, boards: List[chess.Board]):
        """
        Predicts the move.
        """
        output = prediction_utils.compute_move_prediction(self.model, boards)
        return output

    def predict(self, board: chess.Board):
        """
        Predicts the move.
        """
        return self([board])[0]
