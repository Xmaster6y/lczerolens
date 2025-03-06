"""Generic lens class."""

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Generator, Callable, Type, Union, Optional, Any

import torch
import re

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

    def __init__(self, pattern: Optional[str] = None):
        """Initialise the lens.

        Parameters
        ----------
        pattern : Optional[str], default=None
            The pattern to match the modules.
        """
        if pattern is None:
            pattern = r"a^"  # match nothing by default
        self._pattern = pattern
        self._reg_exp = re.compile(pattern)

    @property
    def pattern(self) -> str:
        """The pattern to match the modules."""
        return self._pattern

    @pattern.setter
    def pattern(self, pattern: str):
        self._pattern = pattern
        self._reg_exp = re.compile(pattern)

    def _get_modules(self, model: LczeroModel) -> Generator[tuple[str, Any], None, None]:
        """Get the modules to intervene on."""
        for name, module in model.named_modules():
            fixed_name = name.lstrip(". ")  # nnsight outputs names with a dot
            if self._reg_exp.match(fixed_name):
                yield fixed_name, module

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

    def _trace(
        self,
        model: LczeroModel,
        *inputs: Union[LczeroBoard, torch.Tensor],
        model_kwargs: dict,
        intervention_kwargs: dict,
    ):
        """Trace the model and intervene on it.

        Parameters
        ----------
        model : LczeroModel
            The NNsight model.
        inputs : Union[LczeroBoard, torch.Tensor]
            The inputs.
        model_kwargs : dict
            The model kwargs.
        intervention_kwargs : dict
            The intervention kwargs.

        Returns
        -------
        dict
            The intervention results.
        """
        with model.trace(*inputs, **model_kwargs):
            return self._intervene(model, **intervention_kwargs)

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
        return self._trace(prepared_model, *inputs, model_kwargs=model_kwargs, intervention_kwargs=kwargs)

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
        iter_inputs : Iterable[Tuple[Union[LczeroBoard, torch.Tensor], dict]]
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
        for inputs, dynamic_intervention_kwargs in iter_inputs:
            kwargs.update(dynamic_intervention_kwargs)
            yield self._trace(prepared_model, *inputs, model_kwargs=model_kwargs, intervention_kwargs=kwargs)
