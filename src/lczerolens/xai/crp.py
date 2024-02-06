"""
Compute LRP heatmap for a given model and input.
"""

import chess
import torch
from crp.attribution import CondAttribution
from crp.helper import get_layer_names
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.rules import Epsilon, Flat, Pass, ZPlus
from zennit.types import Activation, Convolution
from zennit.types import Linear as AnyLinear

from lczerolens import board_utils
from lczerolens.adapt.senet import SeNet
from lczerolens.adapt.wrapper import ModelWrapper, PolicyFlow
from lczerolens.game.dataset import GameDataset

from .lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CrpLens(Lens):
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
        return self._compute_crp_heatmap(
            wrapper.model, board, first_map_flat=first_map_flat
        )

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
        raise NotImplementedError

    def _compute_crp_heatmap(
        self,
        model,
        board: chess.Board,
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
        policy_model = PolicyFlow(model)
        layer_names = get_layer_names(
            policy_model, [torch.nn.Conv2d, torch.nn.Linear]
        )
        attribution = CondAttribution(policy_model)
        board_tensor = (
            board_utils.board_to_tensor112x8x8(board)
            .to(DEVICE)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        conditions = [
            {
                "model.block4.conv1": [0],
                "model.block3.conv1": [0],
                "model.block2.conv1": [0],
                "model.block1.conv1": [0],
                "y": list(range(1858)),
            }
        ]
        attr = attribution(
            board_tensor,
            conditions,
            composite,
            record_layer=layer_names,
        )

        heatmap = attr.heatmap
        return heatmap
