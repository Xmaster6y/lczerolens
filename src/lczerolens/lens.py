"""Generic lens class."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Generator, Callable, Type

from lczerolens.model import LczeroModel


class Lens(ABC):
    """Generic lens class for analysing model activations."""

    _registry: Dict[str, Type["Lens"]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Registers the lens.

        Parameters
        ----------
        name : str
            The name of the lens.

        Returns
        -------
        Callable
            The decorator to register the lens.

        Raises
        ------
        ValueError
            If the lens name is already registered.
        """

        if name in cls._registry:
            raise ValueError(f"Lens {name} already registered.")

        def decorator(subclass):
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str, *args, **kwargs) -> "Lens":
        """Returns the lens from its name.

        Parameters
        ----------
        name : str
            The name of the lens.

        Returns
        -------
        Lens
            The lens instance.

        Raises
        ------
        KeyError
            If the lens name is not found.
        """
        if name not in cls._registry:
            raise KeyError(f"Lens {name} not found.")
        return cls._registry[name](*args, **kwargs)

    @abstractmethod
    def is_compatible(self, model: LczeroModel) -> bool:
        """Returns whether the lens is compatible with the model.

        Parameters
        ----------
        model : LczeroModel
            The NNsight model.

        Returns
        -------
        bool
            Whether the lens is compatible with the model.
        """
        pass

    @abstractmethod
    def analyse(
        self,
        *inputs: Any,
        model: LczeroModel,
        **kwargs,
    ) -> Any:
        """Analyse the input.

        Parameters
        ----------
        inputs : Any
            The inputs.
        model : LczeroModel
            The NNsight model.

        Returns
        -------
        Any
            The output.
        """
        pass

    def analyse_batched(
        self,
        iter_inputs: Iterable,
        model: LczeroModel,
        **kwargs,
    ) -> Generator[Any, None, None]:
        """Analyse a batches of inputs.

        Parameters
        ----------
        iter_inputs : Iterator
            The iterator over the inputs.
        model : LczeroModel
            The NNsight model.

        Returns
        -------
        Generator
            The iterator over the statistics.
        """

        for inputs in iter_inputs:
            yield self.analyse(*inputs, model=model, **kwargs)
