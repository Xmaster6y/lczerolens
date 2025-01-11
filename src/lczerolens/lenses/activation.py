"""Activation lens."""

from typing import Optional
import re

import torch

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


@Lens.register("activation")
class ActivationLens(Lens):
    """
    Class for activation-based XAI methods.

    Examples
    --------

        .. code-block:: python

            model = LczeroModel.from_path(model_path)
            lens = ActivationLens()
            board = LczeroBoard()
            results = lens.analyse(board, model=model)
    """

    def __init__(self, pattern: Optional[str] = None):
        if pattern is None:
            pattern = r".*\d+$"
        self._reg_exp = re.compile(pattern)

    def _get_modules(self, model: torch.nn.Module):
        """Get the modules to intervene on."""
        for name, module in model.named_modules():
            if self._reg_exp.match(name):
                yield name, module

    def _intervene(
        self,
        model: LczeroModel,
        **kwargs,
    ) -> dict:
        storage = {}
        for name, module in self._get_modules(model):
            storage[name] = module.output.save()
        return storage
