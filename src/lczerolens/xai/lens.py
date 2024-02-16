"""Generic lens class.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import chess
from torch.utils.data import Dataset

from lczerolens.adapt.wrapper import ModelWrapper


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
    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> Any:
        """
        Computes the heatmap for a given board.
        """
        pass

    @abstractmethod
    def analyse_dataset(
        self,
        dataset: Dataset,
        wrapper: ModelWrapper,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        save_to: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[int, Any]]:
        """
        Computes the statistics for a given board.
        """
        pass
