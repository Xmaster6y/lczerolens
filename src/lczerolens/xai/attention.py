"""
Compute attention heatmap for a given model and input.
"""
from typing import Any, Dict, Optional

import chess

from lczerolens import prediction_utils
from lczerolens.adapt import ModelWrapper


class AttentionWrapper(ModelWrapper):
    """
    Class for wrapping the LCZero models with attention.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the wrapper.
        """
        super().__init__(*args, **kwargs)
        self.attention_cache: Optional[Dict[str, Any]] = None
        self.num_attention_layers = None

    def ensure_has_attention(self):
        """
        Ensures that the model has attention.
        """
        module_name = "Softmax_/encoder0/mha/QK/softmax"
        module_names = [name for name, _ in self.model.named_modules()]
        if module_name not in module_names:
            raise ValueError(
                f"Model at {self.model_path} does not have attention."
            )
        num_layers = 1
        while True:
            module_name = f"Softmax_/encoder{num_layers}/mha/QK/softmax"
            if module_name not in module_names:
                break
            num_layers += 1
        self.num_attention_layers = num_layers

    def cache_attention(self, board: chess.Board, overwrite: bool = False):
        """
        Caches the attention for a given board.
        """
        if self.attention_cache is None or overwrite:
            self.attention_cache = {}
        else:
            raise ValueError("Cache already exists.")
        self.ensure_has_attention()

        removable_handles = []
        named_modules = dict(self.model.named_modules())  # type: ignore
        for layer in range(self.num_attention_layers):
            module_name = f"Softmax_/encoder{layer}/mha/QK/softmax"
            module = named_modules[module_name]

            def cache_hook(module, input, output, module_name=module_name):
                self.attention_cache[module_name] = output

            removable_handles.append(module.register_forward_hook(cache_hook))
        prediction_utils.compute_move_prediction(self.model, [board])
        for handle in removable_handles:
            handle.remove()

    def get_attention(self, attention_layer: int):
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


def compute_attention_heatmap(
    board: chess.Board,
    wrapper: AttentionWrapper,
    attention_layer: int,
):
    """
    Compute attention heatmap for a given model and input.
    """
    wrapper.cache_attention(board, overwrite=True)
    attention = wrapper.get_attention(attention_layer)
    return attention
