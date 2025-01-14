"""Composite lens for XAI."""

from typing import List, Dict, Union, Any

from lczerolens.lens import Lens
from lczerolens.model import LczeroModel


class CompositeLens(Lens):
    """Composite lens for XAI.

    Examples
    --------

        .. code-block:: python

            model = LczeroModel.from_path(model_path)
            lens = CompositeLens([ActivationLens(), GradientLens()])
            board = LczeroBoard()
            results = lens.analyse(board, model=model)
    """

    def __init__(self, lenses: Union[List[Lens], Dict[str, Lens]], merge_results: bool = True):
        self._lens_map = lenses if isinstance(lenses, dict) else {f"lens_{i}": lens for i, lens in enumerate(lenses)}
        self.merge_results = merge_results

    def is_compatible(self, model: LczeroModel) -> bool:
        return all(lens.is_compatible(model) for lens in self._lens_map.values())

    def prepare(self, model: LczeroModel, **kwargs) -> LczeroModel:
        for lens in self._lens_map.values():
            model = lens.prepare(model, **kwargs)
        return model

    def _intervene(self, model: LczeroModel, **kwargs) -> Dict[str, Any]:
        results = {name: lens._intervene(model, **kwargs) for name, lens in self._lens_map.items()}
        if self.merge_results:
            return {k: v for d in results.values() for k, v in d.items()}
        return results
