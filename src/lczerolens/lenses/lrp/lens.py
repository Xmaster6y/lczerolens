"""Compute LRP heatmap for a given model and input."""

from typing import Any


from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


@Lens.register("lrp")
class LrpLens(Lens):
    """Class for wrapping the LCZero models."""

    def _intervene(
        self,
        model: LczeroModel,
        **kwargs,
    ) -> Any:
        # TODO: Refactor this logic
        pass
