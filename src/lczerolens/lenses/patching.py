"""Patching lens."""

from typing import Callable

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

    def __init__(self, patch_fn: Callable, **kwargs):
        self._patch_fn = patch_fn
        super().__init__(**kwargs)

    def _intervene(
        self,
        model: LczeroModel,
        **kwargs,
    ) -> dict:
        for name, module in self._get_modules(model):
            self._patch_fn(name, module, **kwargs)
        return {}
