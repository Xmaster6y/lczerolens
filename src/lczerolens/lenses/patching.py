"""Patching lens for XAI."""

from typing import Callable, Dict

import chess
import torch

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens, LensFactory


@LensFactory.register("patching")
class PatchingLens(Lens):
    """
    Class for patching-based XAI methods.
    """

    def __init__(self, patching_dict: Dict[str, Callable]):
        self.patching_dict = patching_dict
        self.modify_hooks = {}

    def is_compatible(self, model: LczeroModel) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        return isinstance(model, LczeroModel)

    def analyse(
        self,
        board: chess.Board,
        model: LczeroModel,
        **kwargs,
    ) -> torch.Tensor:
        """
        Analyse a single board with the probing lens.
        """
        for modify_hook in self.modify_hooks.values():
            modify_hook.clear()
            modify_hook.register(model)
        out = model(board)
        for modify_hook in self.modify_hooks.values():
            modify_hook.clear()
        return out
