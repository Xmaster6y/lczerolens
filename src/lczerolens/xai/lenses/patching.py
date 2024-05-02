"""Patching lens for XAI."""

from typing import Callable, Dict, Optional

import chess
import torch
from torch.utils.data import Dataset

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

    def analyse_dataset(
        self,
        dataset: Dataset,
        wrapper: ModelWrapper,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        save_to: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """
        Analyse a dataset with the probing lens.
        """
        if save_to is not None:
            raise NotImplementedError("Saving to file is not implemented.")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        batched_outs = None
        for modify_hook in self.modify_hooks.values():
            modify_hook.clear()
            modify_hook.register(wrapper.model)
        for batch in dataloader:
            _, boards = batch
            (out,) = wrapper.predict(boards)
            if batched_outs is None:
                batched_outs = out
            else:
                batched_outs = torch.cat([batched_outs, out])
        for modify_hook in self.modify_hooks.values():
            modify_hook.clear()
        return batched_outs  # type: ignore
