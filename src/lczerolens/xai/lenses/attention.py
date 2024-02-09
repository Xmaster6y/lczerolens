"""
Compute attention heatmap for a given model and input.
"""

from typing import Union

import chess
import torch

from lczerolens.adapt.vitnet import VitNet
from lczerolens.adapt.wrapper import ModelWrapper
from lczerolens.game.dataset import GameDataset
from lczerolens.xai.hook import CacheHook, HookConfig
from lczerolens.xai.lens import Lens


class AttentionLens(Lens):
    """
    Class for wrapping the LCZero models with attention.
    """

    def __init__(self) -> None:
        """
        Initializes the wrapper.
        """
        self.hook_config = HookConfig(
            module_exp=(
                r"encoder(?P<layer>\d+)\/mha\/"
                r"(?P<quantity>(QK)|(Q)|(K)|(out)|(QKV))\/"
                r"(?P<func>(softmax)|(reshape)|(transpose)|(matmul)|(scale))"
            )
        )
        self.hook = CacheHook(self.hook_config)
        self.layer_format = "encoder{layer}/mha/{quantity}/{func}"

    def is_compatible(self, wrapper: ModelWrapper):
        """
        Returns whether the lens is compatible with the model.
        """
        if isinstance(wrapper.model, VitNet):
            return True
        else:
            return False

    def compute_heatmap(
        self, board: chess.Board, wrapper: ModelWrapper, **kwargs
    ) -> torch.Tensor:
        """
        Compute attention heatmap for a given model and input.
        """
        self._cache_attention(board, wrapper)
        attention_layer = kwargs.get("attention_layer", 0)
        attention_quantity = kwargs.get("attention_quantity", "QKV")
        attention_func = kwargs.get("attention_func", "softmax")
        attention = self.get_attention(
            layer=attention_layer,
            quantity=attention_quantity,
            func=attention_func,
        )
        return attention

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

    def _cache_attention(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
    ) -> None:
        """
        Caches the attention for a given board.
        """
        self.hook.register(wrapper)
        wrapper.predict(board)
        self.hook.remove()

    def get_attention(
        self, layer: Union[str, int], quantity: str, func: str
    ) -> torch.Tensor:
        """
        Gets the attention for a given layer and head.
        """
        if isinstance(layer, int):
            layer = str(layer)
        layer_name = self.layer_format.format(
            layer=layer, quantity=quantity, func=func
        )
        attention = self.hook.storage[layer_name]
        if attention is None:
            raise ValueError(f"Attention for layer {layer} does not exist.")
        return attention
