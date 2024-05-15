"""Generic lens class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Iterator

import chess

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
        Computes the statistics for a given board.
        """
        pass

    @abstractmethod
    def analyse_batched_boards(
        self,
        iter_boards: Iterator,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> Optional[Dict[Any, Any]]:
        """
        Computes the statistics for batched boards.
        """
        pass
