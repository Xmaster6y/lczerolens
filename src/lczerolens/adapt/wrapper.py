"""Class for wrapping the LCZero models.
"""

from typing import Iterable, Union

import chess
import torch
from torch import nn

from lczerolens.utils import board as board_utils

from .builder import AutoBuilder


class ModelWrapper(nn.Module):
    """Class for wrapping the LCZero models."""

    def __init__(
        self,
        model: nn.Module,
    ):
        """Initializes the wrapper."""
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    @classmethod
    def from_path(cls, model_path: str, native: bool = True):
        """Creates a wrapper from a model path."""
        model = AutoBuilder.build_from_path(model_path, native=native)
        return cls(model)

    def predict(
        self,
        to_pred: Union[chess.Board, Iterable[chess.Board]],
        with_grad: bool = False,
        input_requires_grad: bool = False,
        return_input: bool = False,
    ):
        """Predicts the move."""
        if isinstance(to_pred, chess.Board):
            board_list = [to_pred]
        elif isinstance(to_pred, Iterable):
            board_list = to_pred  # type: ignore
        else:
            raise ValueError("Invalid input type.")

        tensor_list = [
            board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            for board in board_list
        ]
        batched_tensor = torch.cat(tensor_list, dim=0)
        if input_requires_grad:
            batched_tensor.requires_grad = True
        with torch.set_grad_enabled(with_grad):
            out = self.forward(batched_tensor)

        if return_input:
            out["input"] = batched_tensor
        return out


class PolicyFlow(ModelWrapper):
    """Class for isolating the policy flow."""

    def forward(self, x):
        """Forward pass."""
        return self.model(x)["policy"]


class ValueFlow(ModelWrapper):
    """Class for isolating the value flow."""

    def forward(self, x):
        """Forward pass."""
        return self.model(x)["value"]


class WdlFlow(ModelWrapper):
    """Class for isolating the value flow."""

    def forward(self, x):
        """Forward pass."""
        return self.model(x)["wdl"]


class MlhFlow(ModelWrapper):
    """Class for isolating the value flow."""

    def forward(self, x):
        """Forward pass."""
        return self.model(x)["mlh"]
