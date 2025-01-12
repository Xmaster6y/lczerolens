"""Generic lens class."""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Generator, Callable, Type, Union

import torch

from lczerolens.model import LczeroModel
from lczerolens.board import LczeroBoard


class Lens(ABC):
    """Generic lens class for analysing model activations."""

    _lens_type: str
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

        def decorator(subclass: Type["Lens"]):
            subclass._lens_type = name
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
        return isinstance(model, LczeroModel)

    def prepare(self, model: LczeroModel, **kwargs) -> LczeroModel:
        """Prepare the model for the lens.

        Parameters
        ----------
        model : LczeroModel
            The NNsight model.

        Returns
        -------
        LczeroModel
            The prepared model.
        """
        return model

    @abstractmethod
    def _intervene(self, model: LczeroModel, **kwargs) -> dict:
        """Intervene on the model.

        Parameters
        ----------
        model : LczeroModel
            The NNsight model.

        Returns
        -------
        dict
            The intervention results.
        """
        pass

    def analyse(
        self,
        model: LczeroModel,
        *inputs: Union[LczeroBoard, torch.Tensor],
        **kwargs,
    ) -> dict:
        """Analyse the input.

        Parameters
        ----------
        model : LczeroModel
            The NNsight model.
        inputs : Union[LczeroBoard, torch.Tensor]
            The inputs.

        Returns
        -------
        dict
            The analysis results.

        Raises
        ------
        ValueError
            If the lens is not compatible with the model.
        """
        if not self.is_compatible(model):
            raise ValueError(f"Lens {self._lens_type} is not compatible with the model.")
        model_kwargs = kwargs.get("model_kwargs", {})
        prepared_model = self.prepare(model, **kwargs)
        with prepared_model.trace(*inputs, **model_kwargs):
            return self.intervene(prepared_model, **kwargs)

    def analyse_batched(
        self,
        model: LczeroModel,
        iter_inputs: Iterable[Union[LczeroBoard, torch.Tensor]],
        **kwargs,
    ) -> Generator[dict, None, None]:
        """Analyse a batches of inputs.

        Parameters
        ----------
        model : LczeroModel
            The NNsight model.
        iter_inputs : Iterable[Union[LczeroBoard, torch.Tensor]]
            The iterator over the inputs.

        Returns
        -------
        Generator[dict, None, None]
            The iterator over the statistics.

        Raises
        ------
        ValueError
            If the lens is not compatible with the model.
        """
        if not self.is_compatible(model):
            raise ValueError(f"Lens {self._lens_type} is not compatible with the model.")
        model_kwargs = kwargs.get("model_kwargs", {})
        prepared_model = self.prepare(model, **kwargs)
        for inputs in iter_inputs:
            with prepared_model.trace(*inputs, **model_kwargs):
                yield self.intervene(prepared_model, **kwargs)
