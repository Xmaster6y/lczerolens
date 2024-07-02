"""Activation lens for XAI."""

from typing import Any, Optional, Union, Tuple
import re

import chess
import torch

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens, LensFactory


@LensFactory.register("activation")
class ActivationLens(Lens):
    """
    Class for activation-based XAI methods.

    Examples
    --------

        .. code-block:: python

                model = LczeroModel.from_path(model_path)
                lens = ActivationLens()
                board = chess.Board()
                activations, output = lens.analyse(board, model=model, return_output=True)
                print(activations)
                print(output)
    """

    def __init__(self, pattern: Optional[str] = None):
        if pattern is None:
            pattern = r".*\d+$"
        self._reg_exp = re.compile(pattern)
        self._storage = {}

    @property
    def storage(self):
        return self._storage

    def is_compatible(self, model: LczeroModel) -> bool:
        """Caching is compatible with all torch models."""
        return isinstance(model, LczeroModel)

    def _get_modules(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            if self._reg_exp.match(name):
                yield name, module

    def analyse(
        self,
        *inputs: Union[chess.Board, torch.Tensor],
        model: LczeroModel,
        **kwargs,
    ) -> Tuple[Any, ...]:
        """
        Cache the activations for a given model and input.
        """
        return_output = kwargs.get("return_output", False)
        model_kwargs = kwargs.get("model_kwargs", {})
        self._storage = {}

        with model.trace(*inputs, **model_kwargs):
            for name, module in self._get_modules(model):
                self._storage[name] = module.output.save()
            if return_output:
                output = model.output.save()

        return (self._storage, output) if return_output else (self._storage,)
