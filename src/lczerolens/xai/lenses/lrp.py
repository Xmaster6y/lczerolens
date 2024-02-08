"""
Compute LRP heatmap for a given model and input.
"""

from typing import Any, Dict, List

import chess
import torch
from torch.utils.data import DataLoader
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.rules import Epsilon, Flat, Pass, ZPlus
from zennit.types import Activation, Convolution
from zennit.types import Linear as AnyLinear

from lczerolens.adapt.senet import SeNet
from lczerolens.adapt.wrapper import ModelWrapper
from lczerolens.game.dataset import GameDataset
from lczerolens.xai.lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LrpLens(Lens):
    """
    Class for wrapping the LCZero models.
    """

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        if isinstance(wrapper.model, SeNet):
            return True
        else:
            return False

    def compute_heatmap(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """
        Runs basic LRP on the model.
        """
        first_map_flat = kwargs.get("first_map_flat", False)
        relevance = self._compute_lrp_relevance(
            [board], wrapper, first_map_flat=first_map_flat
        )
        return relevance[0]

    def compute_statistics(
        self,
        dataset: GameDataset,
        wrapper: ModelWrapper,
        batch_size: int,
        **kwargs,
    ) -> dict:
        """
        Computes the statistics for a given board.
        """
        first_map_flat = kwargs.get("first_map_flat", False)
        statistics: Dict[str, Dict[int, Any]] = {
            f"relevance_proportion_h{h}": {} for h in range(8)
        }
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=GameDataset.collate_fn_list,
        )
        for batch in dataloader:
            relevance = self._compute_lrp_relevance(
                batch, wrapper, first_map_flat=first_map_flat
            )
            total_relevances = relevance.abs().sum(dim=(1, 2, 3))
            for i, board in enumerate(batch):
                move_idx = len(board.move_stack)
                for h in range(8):
                    stat_key = f"relevance_proportion_h{h}"
                    relevance_proportion = (
                        relevance[i, 13 * h : 13 * (h + 1)].abs().sum()
                        / total_relevances[i]
                    )
                    if move_idx in statistics[stat_key]:
                        statistics[stat_key][move_idx].append(
                            relevance_proportion.item()
                        )
                    else:
                        statistics[stat_key][move_idx] = [
                            relevance_proportion.item()
                        ]
        return statistics

    def _compute_lrp_relevance(
        self,
        boards: List[chess.Board],
        wrapper: ModelWrapper,
        first_map_flat: bool = False,
    ):
        """
        Compute LRP heatmap for a given model and input.
        """

        canonizers = [SequentialMergeBatchNorm()]

        if first_map_flat:
            first_map = [(AnyLinear, Flat)]
        else:
            first_map = []
        layer_map = [
            (Activation, Pass()),
            (Convolution, ZPlus()),
            (AnyLinear, Epsilon(epsilon=1e-6)),
        ]
        composite = SpecialFirstLayerMapComposite(
            layer_map=layer_map, first_map=first_map, canonizers=canonizers
        )
        with composite.context(wrapper) as modified_model:
            output = modified_model.predict(
                boards,
                with_grad=True,
                input_requires_grad=True,
                return_input=True,
            )
            output["policy"].backward(gradient=output["policy"])
        return output["input"].grad
