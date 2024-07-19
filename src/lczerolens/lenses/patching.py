"""Patching lens for XAI."""

from typing import Callable, Any, Optional, Union, Tuple

import re
import chess
import torch

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens, LensFactory


@LensFactory.register("patching")
class PatchingLens(Lens):
    """
    Class for activation-based XAI methods.

    Examples
    --------

        .. code-block:: python

                model = LczeroModel.from_path(model_path)
                lens = PatchingLens()
                board = chess.Board()
                patch_fn = lambda n, m, *kwargs: pass
                output = lens.analyse(board, model=model)
                print(output)
    """

    def __init__(self, pattern: Optional[str] = None):
        if pattern is None:
            pattern = r".*\d+$"
        self._reg_exp = re.compile(pattern)

    def is_compatible(self, model: LczeroModel) -> bool:
        """Patching is compatible with all LczeroModel models."""
        return isinstance(model, LczeroModel)

    def _get_modules(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            if self._reg_exp.match(name):
                yield name, module

    def analyse(
        self,
        *inputs: Union[chess.Board, torch.Tensor],
        model: LczeroModel,
        patch_fn: Callable,
        **kwargs,
    ) -> Tuple[Any, ...]:
        """
        Cache the activations for a given model and input.
        """
        model_kwargs = kwargs.get("model_kwargs", {})

        with model.trace(*inputs, **model_kwargs):
            for name, module in self._get_modules(model):
                patch_fn(name, module, **kwargs)
            output = model.output.save()

        return output

    def patched_model_factory(
        self,
        model: LczeroModel,
        patch_fn: Callable,
        **kwargs,
    ) -> LczeroModel:
        """
        Create a patched model.

        Parameters
        ----------
        model : LczeroModel
            The model to patch.
        patch_fn : Callable
            The patch function.
        kwargs : Dict
            The keyword arguments.

        Returns
        -------
        LczeroModel
            The patched model.
        """
        model_kwargs = kwargs.get("model_kwargs", {})

        def patched_forward(*inputs: Any, **kwargs: Any) -> torch.Tensor:
            model_kwargs.update(kwargs)

            return self.analyse(*inputs, model=model, patch_fn=patch_fn, model_kwargs=model_kwargs)

        return patched_forward
