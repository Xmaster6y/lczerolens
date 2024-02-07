"""
Generic lens class.
"""

from abc import ABC, abstractmethod

import chess
import torch

from lczerolens.adapt.wrapper import ModelWrapper
from lczerolens.game.dataset import GameDataset


class Lens(ABC):
    """
    Generic lens class.
    """

    @abstractmethod
    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        pass

    @abstractmethod
    def compute_heatmap(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """
        Computes the heatmap for a given board.
        """
        pass

    @abstractmethod
    def compute_statistics(
        self,
        dataset: GameDataset,
        wrapper: ModelWrapper,
        batch_size: int,
        **kwargs,
    ) -> dict:
        """
        Computes the statistics for a given board.
        """
        pass
