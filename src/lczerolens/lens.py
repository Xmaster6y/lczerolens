"""Generic lens class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Iterator, Generator, Union

import torch
import chess

from lczerolens.model import LczeroModel


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
    def is_compatible(self, model: LczeroModel) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        pass

    @abstractmethod
    def analyse(
        self,
        *inputs: Union[chess.Board, torch.Tensor],
        model: LczeroModel,
        **kwargs,
    ) -> Any:
        """
        Computes the statistics for a given inputs.
        """
        pass

    def analyse_batched(
        self,
        iter_inputs: Iterator,
        model: LczeroModel,
        **kwargs,
    ) -> Generator:
        """Cache the activations for a given model.

        Parameters
        ----------
        iter_inputs : Iterator
            The iterator over the boards.
        model : LczeroModel
            The model wrapper.

        Returns
        -------
        Iterator
            The iterator over the activations.
        """

        for inputs in iter_inputs:
            yield self.analyse(*inputs, model=model, **kwargs)
