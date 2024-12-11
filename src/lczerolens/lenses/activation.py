"""Activation lens for XAI."""

from typing import Any, Optional, Union, Tuple, Callable
import re
from dataclasses import dataclass

import chess
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens, LensFactory


@LensFactory.register("activation")
class ActivationLens(Lens):
    """
    Class for activation-based XAI methods.

    Examples
    --------

        .. code-block:: python

                model = LczeroModel.from_path(model_path)
                lens = ActivationLens()
                board = chess.Board()
                activations, output = lens.analyse(board, model=model, return_output=True)
                print(activations)
                print(output)
    """

    def __init__(self, pattern: Optional[str] = None):
        if pattern is None:
            pattern = r".*\d+$"
        self._reg_exp = re.compile(pattern)
        self._storage = {}

    @property
    def storage(self):
        return self._storage

    def is_compatible(self, model: LczeroModel) -> bool:
        """Caching is compatible with all torch models."""
        return isinstance(model, LczeroModel)

    def _get_modules(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            if self._reg_exp.match(name):
                yield name, module

    def analyse(
        self,
        *inputs: Union[chess.Board, torch.Tensor],
        model: LczeroModel,
        **kwargs,
    ) -> Tuple[Any, ...]:
        """
        Cache the activations for a given model and input.
        """
        return_output = kwargs.get("return_output", False)
        model_kwargs = kwargs.get("model_kwargs", {})
        self._storage = {}

        with model.trace(*inputs, **model_kwargs):
            for name, module in self._get_modules(model):
                self._storage[name] = module.output.save()
            if return_output:
                output = model.output.save()

        return (self._storage, output) if return_output else (self._storage,)


@dataclass
class ActivationBuffer:
    model: LczeroModel
    dataset: Dataset
    compute_fn: Callable[[Any, LczeroModel], torch.Tensor]
    n_batches_in_buffer: int = 10
    compute_batch_size: int = 64
    train_batch_size: int = 2048
    dataloader_kwargs: Optional[dict] = None
    logger: Optional[Callable] = None

    def __post_init__(self):
        if self.dataloader_kwargs is None:
            self.dataloader_kwargs = {}
        self._buffer = []
        self._remainder = None
        self._make_dataloader_it()

    def _make_dataloader_it(self):
        self._dataloader_it = iter(
            DataLoader(self.dataset, batch_size=self.compute_batch_size, **self.dataloader_kwargs)
        )

    @torch.no_grad
    def _fill_buffer(self):
        if self.logger is not None:
            self.logger.info("Computing activations...")
        self._buffer = []
        while len(self._buffer) < self.n_batches_in_buffer:
            try:
                next_batch = next(self._dataloader_it)
            except StopIteration:
                break
            activations = self.compute_fn(next_batch, self.model)
            self._buffer.append(activations.to("cpu"))
        if not self._buffer:
            raise StopIteration

    def _make_activations_it(self):
        if self._remainder is not None:
            self._buffer.append(self._remainder)
            self._remainder = None
        activations_ds = TensorDataset(torch.cat(self._buffer, dim=0))
        if self.logger is not None:
            self.logger.info(f"Activations dataset of size {len(activations_ds)}")

        self._activations_it = iter(
            DataLoader(
                activations_ds,
                batch_size=self.train_batch_size,
                shuffle=True,
            )
        )

    def __iter__(self):
        self._make_dataloader_it()
        self._fill_buffer()
        self._make_activations_it()
        self._remainder = None
        return self

    def __next__(self):
        try:
            activations = next(self._activations_it)[0]
            if activations.shape[0] < self.train_batch_size:
                self._remainder = activations
                self._fill_buffer()
                self._make_activations_it()
                activations = next(self._activations_it)[0]
            return activations
        except StopIteration:
            try:
                self._fill_buffer()
                self._make_activations_it()
                self.__next__()
            except StopIteration as e:
                if self._remainder is not None:
                    activations = self._remainder
                    self._remainder = None
                    return activations
                raise StopIteration from e
        raise StopIteration
