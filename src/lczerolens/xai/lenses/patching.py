"""Patching lens for XAI."""

from typing import Callable, Dict, Iterator

import chess
import torch

from lczerolens.model.wrapper import ModelWrapper
from lczerolens.xai.hook import HookConfig, ModifyHook
from lczerolens.xai.lens import Lens


@Lens.register("patching")
class PatchingLens(Lens):
    """
    Class for patching-based XAI methods.
    """

    def __init__(self, patching_dict: Dict[str, Callable]):
        self.patching_dict = patching_dict
        self.modify_hooks = {}
        for module_name, patch in self.patching_dict.items():
            self.modify_hooks[module_name] = ModifyHook(HookConfig(module_exp=module_name, data_fn=patch))

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        return isinstance(wrapper.model, torch.nn.Module)

    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """
        Analyse a single board with the probing lens.
        """
        for modify_hook in self.modify_hooks.values():
            modify_hook.clear()
            modify_hook.register(wrapper.model)
        out = wrapper.predict(board)
        for modify_hook in self.modify_hooks.values():
            modify_hook.clear()
        return out

    def analyse_batched_boards(
        self,
        iter_boards: Iterator,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> Iterator:
        """Patching-based XAI methods.

        Parameters
        ----------
        iter_boards : Iterator
            The iterator over the boards.
        wrapper : ModelWrapper
            The model wrapper.

        Returns
        -------
        Iterator
            The iterator over the patched outputs.
        """
        for modify_hook in self.modify_hooks.values():
            modify_hook.clear()
            modify_hook.register(wrapper.model)
        for batch in iter_boards:
            boards, *_ = batch
            (out,) = wrapper.predict(boards)
            yield out
        for modify_hook in self.modify_hooks.values():
            modify_hook.clear()
