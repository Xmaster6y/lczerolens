"""Probing lens."""

from typing import Callable, Optional

import torch
import re

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


@Lens.register("probing")
class ProbingLens(Lens):
    """
    Class for probing-based XAI methods.

    Examples
    --------

        .. code-block:: python

            model = LczeroModel.from_path(model_path)
            lens = ProbingLens(probe)
            board = LczeroBoard()
            results = lens.analyse(board, model=model)
    """

    def __init__(self, probe_fn: Callable, pattern: Optional[str] = None):
        self._probe_fn = probe_fn
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
        return {
            name: self._probe_fn(module.output.save())
            for name, module in self._get_modules(model)
        }
