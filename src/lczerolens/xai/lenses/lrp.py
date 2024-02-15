"""Compute LRP heatmap for a given model and input.
"""

from typing import Any, Dict, List

import chess
import torch
from torch.utils.data import DataLoader
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Pass, ZPlus
from zennit.types import Activation

from lczerolens.adapt.models.senet import SeNet
from lczerolens.adapt.network import SumLayer
from lczerolens.adapt.wrapper import ModelWrapper
from lczerolens.game.dataset import GameDataset
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

    def compute_heatmap(
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
        relevance = self._compute_lrp_relevance(
            [board], wrapper, composite=composite
        )
        return relevance[0]

    def compute_statistics(
        self,
        dataset: GameDataset,
        wrapper: ModelWrapper,
        batch_size: int,
        **kwargs,
    ) -> dict:
        """Computes the statistics for a given board.

        Parameters
        ----------
        dataset : GameDataset
            The dataset to compute the statistics for.
        wrapper : ModelWrapper
            The model wrapper.
        batch_size : int
            The batch size.
        """
        composite = kwargs.get("composite", None)
        statistics: Dict[str, Dict[int, Any]] = {
            "planes_relevance_proportion": {},
            "planes_relevance_proportion_scaled": {},
        }
        for piece_type in range(1, 13):
            statistics.update(
                {
                    (
                        "configuration_relevance_proportion_"
                        f"threatened_piece{piece_type}"
                    ): {},
                }
            )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=GameDataset.collate_fn_list,
        )
        for batch in dataloader:
            relevance = self._compute_lrp_relevance(
                batch, wrapper, composite=composite
            )
            total_relevances = relevance.abs().sum(dim=(1, 2, 3))
            for i, board in enumerate(batch):
                move_idx = len(board.move_stack)
                stat_key = "planes_relevance_proportion"
                relevance_proportion = (
                    relevance[i].abs().sum(dim=(1, 2)) / total_relevances[i]
                )
                if move_idx in statistics[stat_key]:
                    statistics[stat_key][move_idx].append(relevance_proportion)
                else:
                    statistics[stat_key][move_idx] = [relevance_proportion]
                stat_key = "planes_relevance_proportion_scaled"
                n_pieces = (relevance[i] != 0.0).sum(dim=(1, 2))
                n_pieces_or_one = n_pieces + (n_pieces == 0).float()
                relevance_proportion = (
                    relevance[i].abs().sum(dim=(1, 2))
                    / n_pieces_or_one
                    / total_relevances[i]
                )
                if move_idx in statistics[stat_key]:
                    statistics[stat_key][move_idx].append(relevance_proportion)
                else:
                    statistics[stat_key][move_idx] = [relevance_proportion]

                us = board.turn
                them = not us
                configuration_relevance = relevance[i, :12].sum(dim=0).view(64)
                total_configuration_relevance = (
                    configuration_relevance.abs().sum()
                )
                for piece_type in range(1, 7):
                    for player in [us, them]:
                        stat_key = (
                            "configuration_relevance_proportion_"
                            f"threatened_piece{piece_type+6*(1-player)}"
                        )
                    pieces = board.pieces(piece_type, player)
                    for square in pieces:
                        n_attackers = len(board.attackers(not player, square))
                        if us == chess.BLACK:
                            square = chess.square_mirror(square)
                        square_relevance_proportion = (
                            configuration_relevance[square].abs().sum()
                            / total_configuration_relevance
                        )
                        if n_attackers in statistics[stat_key]:
                            statistics[stat_key][n_attackers].append(
                                square_relevance_proportion.item()
                            )
                        else:
                            statistics[stat_key][n_attackers] = [
                                square_relevance_proportion.item()
                            ]
        return statistics

    def _compute_lrp_relevance(
        self,
        boards: List[chess.Board],
        wrapper: ModelWrapper,
        composite: Any = None,
    ):
        """
        Compute LRP heatmap for a given model and input.
        """

        if composite is None:
            composite = self.make_default_composite()

        with composite.context(wrapper) as modified_model:
            output = modified_model.predict(
                boards,
                with_grad=True,
                input_requires_grad=True,
                return_input=True,
            )
            output["policy"].backward(gradient=output["policy"])
        return output["input"].grad

    @staticmethod
    def make_default_composite():
        canonizers = [SequentialMergeBatchNorm()]

        layer_map = [
            (Activation, Pass()),
            (torch.nn.Conv2d, ZPlus()),
            (torch.nn.Linear, Epsilon(epsilon=1e-6)),
            (SumLayer, Epsilon(epsilon=1e-6)),
            (torch.nn.AdaptiveAvgPool2d, Epsilon(epsilon=1e-6)),
        ]
        return LayerMapComposite(layer_map=layer_map, canonizers=canonizers)
