"""Probing lens."""

from typing import Callable

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

    def __init__(self, probe_fn: Callable, **kwargs):
        self._probe_fn = probe_fn
        super().__init__(**kwargs)

    def _intervene(
        self,
        model: LczeroModel,
        **kwargs,
    ) -> dict:
        return {name: self._probe_fn(module.output.save()) for name, module in self._get_modules(model)}
