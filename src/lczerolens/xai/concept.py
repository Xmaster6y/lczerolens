"""
Class for concept-based XAI methods.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import chess

from lczerolens.game.dataset import GameDataset


class ConceptType(str, Enum):
    """
    Enum for concept type.
    """

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    COMPOSITE = "composite"


class Concept(ABC):
    """
    Class for concept-based XAI methods.
    """

    @abstractmethod
    def compute_label(
        self,
        board: chess.Board,
    ) -> Any:
        """
        Compute the label for a given model and input.
        """
        pass


class ConceptDataset(GameDataset):
    """
    Class for concept
    """

    def __init__(self, *args, concept: Concept, **kwargs):
        super().__init__(*args, **kwargs)
        self.concept = concept

    def __getitem__(self, idx) -> chess.Board:
        board = super().__getitem__(idx)
        label = self.concept.compute_label(board)
        return board, label
