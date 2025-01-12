"""Activation lens for XAI."""

from typing import Any, Optional, Callable
from dataclasses import dataclass

import torch
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset

from lczerolens.model import LczeroModel


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
