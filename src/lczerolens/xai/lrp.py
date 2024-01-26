"""
Compute LRP heatmap for a given model and input.
"""

from typing import Any, Dict

import chess
import torch
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.rules import Epsilon, Flat, Pass, ZPlus
from zennit.types import Activation, Convolution
from zennit.types import Linear as AnyLinear

from lczerolens import prediction_utils
from lczerolens.adapt import ModelWrapper

from .lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LrpLens(Lens):
    """
    Class for wrapping the LCZero models.
    """

    def compute_heatmap(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Runs basic LRP on the model.
        """
        relevance = self._compute_lrp_heatmap(wrapper.model, board)
        return relevance

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        return True

    def _compute_lrp_heatmap(
        self, model, board: chess.Board, first_map_flat: bool = False
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
        with composite.context(model) as modified_model:
            output = prediction_utils.compute_move_prediction(
                modified_model,
                [board],
                with_grad=True,
                input_requires_grad=True,
                return_input=True,
            )
            output["policy"].backward(gradient=output["policy"])
        relevance = output["input"].grad[0]
        return relevance
