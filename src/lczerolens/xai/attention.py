"""
Compute attention heatmap for a given model and input.
"""
import re
from typing import Dict, Optional, Union

import chess
import torch

from lczerolens.adapt import ModelWrapper

from .lens import CacheHookArgs, CacheHookFactory, CacheMode, Lens


class AttentionLens(Lens):
    """
    Class for wrapping the LCZero models with attention.
    """

    def __init__(self) -> None:
        """
        Initializes the wrapper.
        """
        self.attention_cache: Optional[Dict[str, torch.Tensor]] = None
        self.hook_factory = CacheHookFactory(mode=CacheMode.OUTPUT)
        self.module_name_exp = (
            r"encoder(?P<layer>\d+)\/mha\/"
            r"(?P<quantity>(QK)|(Q)|(K)|(out)|(QKV))\/"
            r"(?P<func>(softmax)|(reshape)|(transpose)|(matmul)|(scale))"
        )

    def is_compatible(self, wrapper: ModelWrapper):
        """
        Returns whether the lens is compatible with the model.
        """
        return self._has_attention(wrapper)

    def compute_heatmap(
        self, board: chess.Board, wrapper: ModelWrapper, **kwargs
    ) -> torch.Tensor:
        """
        Compute attention heatmap for a given model and input.
        """
        self._cache_attention(board, wrapper, overwrite=True)
        attention_layer = kwargs.get("attention_layer", -1)
        attention = self.get_attention(attention_layer)
        return attention

    def _has_attention(self, wrapper: ModelWrapper) -> bool:
        """
        Ensures that the model has attention.
        """
        for name, _ in wrapper.model.named_modules():
            match = re.search(self.module_name_exp, name)
            if match:
                return True
        return False

    def _cache_attention(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        overwrite: bool = False,
    ) -> None:
        """
        Caches the attention for a given board.
        """
        if self.attention_cache is None or overwrite:
            self.attention_cache = {}
        else:
            raise ValueError("Cache already exists.")

        module_registry = {}
        args_registry = {}
        for name, module in wrapper.model.named_modules():
            match = re.search(self.module_name_exp, name)
            if match:
                key = (
                    f"{match.group('layer')}-"
                    f"{match.group('quantity')}"
                    f"-{match.group('func')}"
                )
                module_registry[name] = module
                args_registry[name] = CacheHookArgs(
                    cache=self.attention_cache,
                    key=key,
                )
        self.hook_factory.register(module_registry, args_registry)
        wrapper.predict(board)
        self.hook_factory.remove()

    def get_attention(self, layer: Union[str, int]):
        """
        Gets the attention for a given layer and head.
        """
        if self.attention_cache is None:
            raise ValueError(
                "Cache does not exist. Call cache_attention first."
            )
        if layer == -1:
            return self.attention_cache
        if isinstance(layer, int):
            layer = str(layer)
        attention = self.attention_cache.get(layer, None)
        if attention is None:
            raise ValueError(f"Attention for layer {layer} does not exist.")
        return attention
