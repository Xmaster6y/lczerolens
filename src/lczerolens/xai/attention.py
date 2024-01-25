"""
Compute attention heatmap for a given model and input.
"""
from typing import Any, Dict, Optional

import chess

from lczerolens.adapt import ModelWrapper

from .lens import Lens


class AttentionLens(Lens):
    """
    Class for wrapping the LCZero models with attention.
    """

    def __init__(self):
        """
        Initializes the wrapper.
        """
        self.attention_cache: Optional[Dict[str, Any]] = None

    def is_compatible(self, wrapper: ModelWrapper):
        """
        Returns whether the lens is compatible with the model.
        """
        return self._has_attention(wrapper)

    def compute_heatmap(
        self, board: chess.Board, wrapper: ModelWrapper, **kwargs
    ):
        """
        Compute attention heatmap for a given model and input.
        """
        self._cache_attention(board, wrapper, overwrite=True)
        attention_layer = kwargs.get("attention_layer", -1)
        attention = self._get_attention(attention_layer)
        return attention

    def _has_attention(self, wrapper: ModelWrapper) -> bool:
        """
        Ensures that the model has attention.
        """
        module_name = "Softmax_/encoder0/mha/QK/softmax"
        module_names = [name for name, _ in wrapper.model.named_modules()]
        if module_name not in module_names:
            return False
        return True

    def _cache_attention(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        overwrite: bool = False,
    ):
        """
        Caches the attention for a given board.
        """
        if self.attention_cache is None or overwrite:
            self.attention_cache = {}
        else:
            raise ValueError("Cache already exists.")

        removable_handles = []
        named_modules = dict(wrapper.model.named_modules())  # type: ignore
        layer = 0
        module_name = f"Softmax_/encoder{layer}/mha/QK/softmax"
        while module_name in named_modules:
            module = named_modules[module_name]

            def cache_hook(module, input, output, module_name=module_name):
                self.attention_cache[module_name] = output

            removable_handles.append(module.register_forward_hook(cache_hook))
            layer += 1
        wrapper.predict(board)
        for handle in removable_handles:
            handle.remove()

    def _get_attention(self, attention_layer: int):
        """
        Gets the attention for a given layer and head.
        """
        if self.attention_cache is None:
            raise ValueError(
                "Cache does not exist. Call cache_attention first."
            )
        attention = self.attention_cache.get(
            f"Softmax_/encoder{attention_layer}/mha/QK/softmax", None
        )
        if attention is None:
            raise ValueError(
                f"Attention for layer {attention_layer} does not exist."
            )
        return attention
