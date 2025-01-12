"""Composite lens for XAI."""

from typing import List

from lczerolens.lens import Lens
from lczerolens.model import LczeroModel


class CompositeLens(Lens):
    """Composite lens for XAI.

    Examples
    --------

        .. code-block:: python

            model = LczeroModel.from_path(model_path)
            lens = CompositeLens([ActivationLens(), OutputLens()])
            board = LczeroBoard()
            results = lens.analyse(board, model=model)
    """

    def __init__(self, lenses: List[Lens]):
        self.lenses = lenses

    def is_compatible(self, model: LczeroModel) -> bool:
        return all(lens.is_compatible(model) for lens in self.lenses)

    def prepare(self, model: LczeroModel, **kwargs) -> LczeroModel:
        for lens in self.lenses:
            model = lens.prepare(model, **kwargs)
        return model

    def _intervene(self, model: LczeroModel, **kwargs) -> dict:
        return {f"lens_{i}": lens._intervene(model, **kwargs) for i, lens in enumerate(self.lenses)}
