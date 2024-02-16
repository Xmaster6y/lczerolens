"""Activation lens for XAI.
"""

from typing import Any, Callable, Dict, Optional

import chess
import torch
from torch.utils.data import DataLoader, Dataset

from lczerolens.adapt.wrapper import ModelWrapper
from lczerolens.xai.hook import CacheHook, HookConfig
from lczerolens.xai.lens import Lens


class ActivationLens(Lens):
    """Class for activation-based XAI methods."""

    def __init__(self, module_exp: Optional[str] = None):
        if module_exp is None:
            module_exp = r"block\d+$"
        self._module_exp = module_exp
        self._cache_hook = CacheHook(HookConfig(module_exp=module_exp))

    @property
    def module_exp(self):
        return self._module_exp

    @module_exp.setter
    def module_exp(self, value):
        self._module_exp = value
        self.cache_hook = CacheHook(HookConfig(module_exp=value))

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """Caching is compatible with all torch models."""
        return isinstance(wrapper.model, torch.nn.Module)

    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> Any:
        """Cache the activations for a given model and input."""
        self.cache_hook.clear()
        self.cache_hook.register(wrapper.model)
        wrapper.predict(board)
        return self.cache_hook.storage.copy()

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
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        self.cache_hook.clear()
        self.cache_hook.register(wrapper.model)
        cache: Dict[int, Any] = {}
        for batch in dataloader:
            inidices, boards = batch
            wrapper.predict(boards)
            for idx in inidices:
                cache[idx] = {}
                for key, value in self.cache_hook.storage.items():
                    cache[idx][key] = value[idx]
        return cache
