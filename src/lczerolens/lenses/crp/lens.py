"""Compute CRP heatmap for a given model and input."""

from typing import Any, List, Optional

import chess
import torch
from crp.attribution import CondAttribution
from crp.helper import get_layer_names

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens

from lczerolens.lenses.lrp.lens import LrpLens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@Lens.register("crp")
class CrpLens(Lens):
    """
    Class for wrapping the LCZero models.
    """

    def is_compatible(self, model: LczeroModel) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        return isinstance(model.model, torch.nn.Module)

    def analyse_board(
        self,
        board: chess.Board,
        model: LczeroModel,
        **kwargs,
    ) -> torch.Tensor:
        mode = kwargs.get("mode", "latent_relevances")
        layer_names = kwargs.get("layer_names", None)
        composite = kwargs.get("composite", None)

        if mode == "latent_relevances":
            return self._compute_latent_relevances([board], model, layer_names=layer_names, composite=composite)
        elif mode == "max_ref":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid mode {mode}")

    def _compute_latent_relevances(
        self,
        boards: List[chess.Board],
        model: LczeroModel,
        layer_names: Optional[List[str]] = None,
        composite: Optional[Any] = None,
    ) -> torch.Tensor:
        if layer_names is None:
            layer_names = layer_names = get_layer_names(model, [torch.nn.Identity])
        if composite is None:
            composite = LrpLens.make_default_composite()

        attribution = CondAttribution(model)

        attr = attribution(boards, composite, record_layer=layer_names)
        return attr.relevances
