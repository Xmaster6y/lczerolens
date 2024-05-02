"""Compute CRP heatmap for a given model and input."""

from typing import Any, Callable, List, Optional

import chess
import torch
from crp.attribution import CondAttribution
from crp.helper import get_layer_names
from torch.utils.data import Dataset

from lczerolens.model.wrapper import ModelWrapper
from lczerolens.xai.lens import Lens

from .lrp import LrpLens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@Lens.register("crp")
class CrpLens(Lens):
    """
    Class for wrapping the LCZero models.
    """

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        return isinstance(wrapper.model, torch.nn.Module)

    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        mode = kwargs.get("mode", "latent_relevances")
        layer_names = kwargs.get("layer_names", None)
        composite = kwargs.get("composite", None)

        if mode == "latent_relevances":
            return self._compute_latent_relevances([board], wrapper, layer_names=layer_names, composite=composite)
        elif mode == "max_ref":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid mode {mode}")

    def analyse_dataset(
        self,
        dataset: Dataset,
        wrapper: ModelWrapper,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        save_to: Optional[str] = None,
        **kwargs,
    ) -> dict:
        raise NotImplementedError

    def _compute_latent_relevances(
        self,
        boards: List[chess.Board],
        wrapper: ModelWrapper,
        layer_names: Optional[List[str]] = None,
        composite: Optional[Any] = None,
    ) -> torch.Tensor:
        if layer_names is None:
            layer_names = layer_names = get_layer_names(wrapper, [torch.nn.Identity])
        if composite is None:
            composite = LrpLens.make_default_composite()

        attribution = CondAttribution(wrapper)

        attr = attribution(boards, composite, record_layer=layer_names)
        return attr.relevances
