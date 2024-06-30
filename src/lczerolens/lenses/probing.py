"""Probing lens for XAI."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import einops
import torch
import chess

from lczerolens.model import LczeroModel
from lczerolens.lens import Lens


EPS = 1e-6


class Probe(ABC):
    """Abstract class for probes."""

    def __init__(self):
        self._trained = False

    @abstractmethod
    def train(
        self,
        activations: torch.Tensor,
        labels: Any,
        **kwargs,
    ):
        """Train the probe."""
        pass

    @abstractmethod
    def predict(self, activations: torch.Tensor, **kwargs):
        """Predict with the probe."""
        pass


class SignalCav(Probe):
    """Signal CAV probe."""

    def train(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ):
        if len(activations) != len(labels):
            raise ValueError("Number of activations and labels must match")
        if len(activations.shape) != 2:
            raise ValueError("Activations must a batch of tensors")
        if len(labels.shape) != 2:
            raise ValueError("Labels must a batch of tensors")

        mean_activation = activations.mean(dim=1, keepdim=True)
        mean_label = labels.mean(dim=1, keepdim=True)
        scaled_activations = activations - mean_activation
        scaled_labels = labels - mean_label
        cav = einops.einsum(scaled_activations, scaled_labels, "b a, b d -> a d")
        self._h = cav / (cav.norm(dim=0, keepdim=True) + EPS)

    def predict(self, activations: torch.Tensor, **kwargs):
        if not self._trained:
            raise ValueError("Probe not trained")

        if len(activations.shape) != 2:
            raise ValueError("Activations must a batch of tensors")

        dot_prod = einops.einsum(activations, self._h, "b a, a d -> b d")
        return dot_prod / (activations.norm(dim=1, keepdim=True) + EPS)


@Lens.register("probing")
class ProbingLens(Lens):
    """
    Class for probing-based XAI methods.
    """

    def __init__(self, probe_dict: Dict[str, Probe]):
        self.probe_dict = probe_dict
        self.measure_hooks = {}

    def is_compatible(self, model: LczeroModel) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        return isinstance(model.model, torch.nn.Module)

    def analyse_board(
        self,
        board: chess.Board,
        model: LczeroModel,
        **kwargs,
    ) -> torch.Tensor:
        """
        Analyse a single board with the probing lens.
        """
        measures = {}
        for measure_hook in self.measure_hooks.values():
            measure_hook.clear()
            measure_hook.register(model.model)
        model.predict(board)
        for measure_hook in self.measure_hooks.values():
            measures.update(measure_hook.storage)
            measure_hook.clear()
        return measures
