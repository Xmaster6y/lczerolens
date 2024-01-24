"""
Class for wrapping the LCZero models.
"""

from typing import List

import chess

from lczerolens import prediction_utils

from .builder import AutoBuilder, LczeroModel


class ModelWrapper:
    """
    Class for wrapping the LCZero models.
    """

    model_builder = AutoBuilder

    def __init__(
        self,
        model: LczeroModel,
    ):
        """
        Initializes the wrapper.
        """
        self.model = model

    @classmethod
    def from_path(cls, model_path):
        """
        Creates a wrapper from a model path.
        """
        lczero_model = cls.model_builder.build_from_path(model_path)
        return cls(lczero_model)

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
