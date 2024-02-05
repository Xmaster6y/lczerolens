"""
Class for wrapping the LCZero models.
"""

from typing import List, Union

import chess
from torch import nn

from lczerolens.utils import prediction as prediction_utils

from .builder import AutoBuilder


class ModelWrapper(nn.Module):
    """
    Class for wrapping the LCZero models.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        """
        Initializes the wrapper.
        """
        super().__init__()
        self.model = model

    def forward(self, x):
        """
        Forward pass.
        """
        return self.model(x)

    @classmethod
    def from_path(cls, model_path):
        """
        Creates a wrapper from a model path.
        """
        model = AutoBuilder.build_from_path(model_path)
        return cls(model)

    def predict(self, to_pred: Union[chess.Board, List[chess.Board]]):
        """
        Predicts the move.
        """
        if isinstance(to_pred, chess.Board):
            output = prediction_utils.compute_move_prediction(self, [to_pred])
            return output[0]
        elif isinstance(to_pred, list):
            output = prediction_utils.compute_move_prediction(self, to_pred)
            return output
        else:
            raise ValueError("Invalid input type")


class PolicyFlow(ModelWrapper):
    """
    Class for isolating the policy flow.
    """

    def forward(self, x):
        """
        Forward pass.
        """
        return self.model(x)["policy"]


class ValueFlow(ModelWrapper):
    """
    Class for isolating the value flow.
    """

    def forward(self, x):
        """
        Forward pass.
        """
        return self.model(x)["value"]


class WdlFlow(ModelWrapper):
    """
    Class for isolating the value flow.
    """

    def forward(self, x):
        """
        Forward pass.
        """
        return self.model(x)["wdl"]


class MlhFlow(ModelWrapper):
    """
    Class for isolating the value flow.
    """

    def forward(self, x):
        """
        Forward pass.
        """
        return self.model(x)["mlh"]
