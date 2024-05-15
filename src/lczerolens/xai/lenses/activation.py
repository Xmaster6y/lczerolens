"""Activation lens for XAI."""

from typing import Any, Optional, Iterator
import copy

import chess
import torch

from lczerolens.model.wrapper import ModelWrapper
from lczerolens.xai.hook import CacheHook, HookConfig
from lczerolens.xai.lens import Lens


@Lens.register("activation")
class ActivationLens(Lens):
    """Class for activation-based XAI methods."""

    def __init__(self, module_exp: Optional[str] = None):
        if module_exp is None:
            module_exp = r"block\d+$"
        self.cache_hook = CacheHook(HookConfig(module_exp=module_exp))

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
        return copy.deepcopy(self.cache_hook.storage)

    def analyse_batched_boards(
        self,
        iter_boards: Iterator,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> Iterator:
        """Cache the activations for a given model.

        Parameters
        ----------
        iter_boards : Iterator
            The iterator over the boards.
        wrapper : ModelWrapper
            The model wrapper.

        Returns
        -------
        Iterator
            The iterator over the activations.
        """
        self.cache_hook.clear()
        self.cache_hook.register(wrapper.model)
        for batch in iter_boards:
            boards, *_ = batch
            wrapper.predict(boards)
            yield copy.deepcopy(self.cache_hook.storage)
        self.cache_hook.clear()
