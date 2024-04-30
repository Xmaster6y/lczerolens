"""Generic lens class."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Type

import chess
from torch.utils.data import Dataset

from lczerolens.model.wrapper import ModelWrapper


class Lens(ABC):
    """
    Generic lens class.
    """

    all_lenses: Dict[str, Type["Lens"]] = {}

    @classmethod
    def register(cls, name: str):
        """Registers the lens."""

        def decorator(subclass):
            cls.all_lenses[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "Lens":
        """Returns the lens from its name."""
        return cls.all_lenses[name](**kwargs)

    @classmethod
    def get_subclass(cls, name: str) -> Type["Lens"]:
        """Returns the subclass."""
        return cls.all_lenses[name]

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
    ) -> Optional[Dict[Any, Any]]:
        """
        Computes the statistics for a given board.
        """
        pass
