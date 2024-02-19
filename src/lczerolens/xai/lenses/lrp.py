"""Compute LRP heatmap for a given model and input.
"""

from typing import Any, Callable, Dict, List, Optional

import chess
import torch
from torch.utils.data import DataLoader, Dataset
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Pass, ZPlus
from zennit.types import Activation

from lczerolens.adapt.models.senet import SeNet
from lczerolens.adapt.network import ProdLayer, SumLayer
from lczerolens.adapt.wrapper import ModelWrapper
from lczerolens.xai.lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LrpLens(Lens):
    """Class for wrapping the LCZero models."""

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """Returns whether the lens is compatible with the model.

        Parameters
        ----------
        wrapper : ModelWrapper
            The model wrapper.

        Returns
        -------
        bool
            Whether the lens is compatible with the model.
        """
        if isinstance(wrapper.model, SeNet):
            return True
        else:
            return False

    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """Runs basic LRP on the model.

        Parameters
        ----------
        board : chess.Board
            The board to compute the heatmap for.
        wrapper : ModelWrapper
            The model wrapper.

        Returns
        -------
        torch.Tensor
            The heatmap for the given board.
        """
        composite = kwargs.get("composite", None)
        target = kwargs.get("target", "policy")
        relevance = self._compute_lrp_relevance(
            [board], wrapper, composite=composite, target=target
        )
        return relevance[0]

    def analyse_dataset(
        self,
        dataset: Dataset,
        wrapper: ModelWrapper,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        save_to: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[int, Any]]:
        """Cache the activations for a given model and dataset."""
        if save_to is not None:
            raise NotImplementedError("Saving to file is not implemented.")
        composite = kwargs.get("composite", None)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        relevances = {}
        for batch in dataloader:
            inidices, boards = batch
            batched_relevances = self._compute_lrp_relevance(
                boards, wrapper, composite=composite
            )
            for idx, relevance in zip(inidices, batched_relevances):
                relevances[idx] = relevance
        return relevances

    def _compute_lrp_relevance(
        self,
        boards: List[chess.Board],
        wrapper: ModelWrapper,
        composite: Optional[Any] = None,
        target: Optional[str] = None,
    ):
        """
        Compute LRP heatmap for a given model and input.
        """

        if composite is None:
            composite = self.make_default_composite()

        with composite.context(wrapper) as modified_model:
            output, input_tensor = modified_model.predict(
                boards,
                with_grad=True,
                input_requires_grad=True,
                return_input=True,
            )
            if target is None:
                output.backward(gradient=output)
            else:
                output[target].backward(gradient=output[target])
        return input_tensor.grad

    @staticmethod
    def make_default_composite():
        canonizers = [SequentialMergeBatchNorm()]

        layer_map = [
            (Activation, Pass()),
            (torch.nn.Conv2d, ZPlus()),
            (torch.nn.Linear, Epsilon(epsilon=1e-6)),
            (SumLayer, Epsilon(epsilon=1e-6)),
            (ProdLayer, Epsilon(epsilon=1e-6)),
            (torch.nn.AdaptiveAvgPool2d, Epsilon(epsilon=1e-6)),
        ]
        return LayerMapComposite(layer_map=layer_map, canonizers=canonizers)
