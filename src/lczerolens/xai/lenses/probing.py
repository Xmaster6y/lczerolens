"""Probing lens for XAI.
"""

from typing import Callable, Optional

import chess
import torch
from torch.utils.data import Dataset

from lczerolens.game.wrapper import ModelWrapper
from lczerolens.xai.lens import Lens


@Lens.register("probing")
class ProbingLens(Lens):
    """
    Class for probing-based XAI methods.
    """

    def __init__(self, probe):
        self.probe = probe

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        if hasattr(wrapper.model, "block0"):
            return True
        else:
            return False

    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def analyse_dataset(
        self,
        dataset: Dataset,
        wrapper: ModelWrapper,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        save_to: Optional[str] = None,
        **kwargs,
    ) -> dict:
        raise NotImplementedError
