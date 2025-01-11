"""Patching lens."""

from typing import Callable, Optional

import re
import torch

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


@Lens.register("patching")
class PatchingLens(Lens):
    """
    Class for activation-based XAI methods.

    Examples
    --------

        .. code-block:: python

            model = LczeroModel.from_path(model_path)
            lens = PatchingLens()
            board = LczeroBoard()
            patch_fn = lambda n, m, *kwargs: pass
            results = lens.analyse(board, model=model)
    """

    def __init__(self, patch_fn: Callable, pattern: Optional[str] = None):
        self._patch_fn = patch_fn
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
        for name, module in self._get_modules(model):
            self._patch_fn(name, module, **kwargs)
        return {}
