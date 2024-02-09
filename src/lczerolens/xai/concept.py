"""Class for concept-based XAI methods.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import chess
import torch
from sklearn import metrics

from lczerolens.game.dataset import GameDataset
from lczerolens.utils import board as board_utils


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

    @staticmethod
    @abstractmethod
    def compute_metrics(
        predictions,
        labels,
    ):
        """
        Compute the metrics for a given model and input.
        """
        pass


class BinaryConcept(Concept):
    """
    Class for binary concept-based XAI methods.
    """

    @staticmethod
    def compute_metrics(
        predictions,
        labels,
    ):
        """
        Compute the metrics for a given model and input.
        """
        return {
            "accuracy": metrics.accuracy_score(labels, predictions),
            "precision": metrics.precision_score(labels, predictions),
            "recall": metrics.recall_score(labels, predictions),
            "f1": metrics.f1_score(labels, predictions),
        }


class OrBinaryConcept(BinaryConcept):
    """
    Class for binary concept-based XAI methods.
    """

    def __init__(self, *concepts: BinaryConcept):
        for concept in concepts:
            if not isinstance(concept, BinaryConcept):
                raise ValueError(f"{concept} is not a BinaryConcept")
        self.concepts = concepts

    def compute_label(
        self,
        board: chess.Board,
    ) -> Any:
        """
        Compute the label for a given model and input.
        """
        return any(concept.compute_label(board) for concept in self.concepts)


class AndBinaryConcept(BinaryConcept):
    """
    Class for binary concept-based XAI methods.
    """

    def __init__(self, *concepts: BinaryConcept):
        for concept in concepts:
            if not isinstance(concept, BinaryConcept):
                raise ValueError(f"{concept} is not a BinaryConcept")
        self.concepts = concepts

    def compute_label(
        self,
        board: chess.Board,
    ) -> Any:
        """
        Compute the label for a given model and input.
        """
        return all(concept.compute_label(board) for concept in self.concepts)


class MulticlassConcept(Concept):
    """
    Class for multiclass concept-based XAI methods.
    """

    @staticmethod
    def compute_metrics(
        predictions,
        labels,
    ):
        """
        Compute the metrics for a given model and input.
        """
        return {
            "accuracy": metrics.accuracy_score(labels, predictions),
            "precision": metrics.precision_score(
                labels, predictions, average="weighted"
            ),
            "recall": metrics.recall_score(
                labels, predictions, average="weighted"
            ),
            "f1": metrics.f1_score(labels, predictions, average="weighted"),
        }


class ContinuousConcept(Concept):
    """
    Class for continuous concept-based XAI methods.
    """

    @staticmethod
    def compute_metrics(
        predictions,
        labels,
    ):
        """
        Compute the metrics for a given model and input.
        """
        return {
            "rmse": metrics.root_mean_squared_error(labels, predictions),
            "mae": metrics.mean_absolute_error(labels, predictions),
            "r2": metrics.r2_score(labels, predictions),
        }


class ConceptDataset(GameDataset):
    """
    Class for concept
    """

    def __init__(self, concept: Concept, file_name: Optional[str]):
        super().__init__(file_name=file_name)
        self.concept = concept

    def __getitem__(self, idx) -> chess.Board:
        board = super().__getitem__(idx)
        label = self.concept.compute_label(board)
        return board, label

    @classmethod
    def from_game_dataset(cls, game_dataset: GameDataset, concept: Concept):
        instance = cls(concept=concept, file_name=None)
        instance.games = game_dataset.games
        instance.cache = None
        return instance

    @staticmethod
    def collate_fn_list(batch):
        board_list, label_list = zip(*batch)
        return board_list, label_list

    @staticmethod
    def collate_fn_tensor(batch):
        board_list, label_list = zip(*batch)
        tensor_list = [
            board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            for board in board_list
        ]
        batched_tensor = torch.cat(tensor_list, dim=0)
        return batched_tensor, label_list


class UniqueConceptDataset(ConceptDataset):
    """
    Class for unique board concept
    """

    def __init__(
        self, concept: Concept, file_name: Optional[str], strict: bool = False
    ):
        super().__init__(concept, file_name)
        self._unique_resample(strict=strict)

    def __getitem__(self, idx) -> chess.Board:
        board = self.unique_boards[idx]
        label = self.concept.compute_label(board)
        return board, label

    def __len__(self):
        return len(self.unique_boards)

    @classmethod
    def from_game_dataset(
        cls, game_dataset: GameDataset, concept: Concept, strict: bool = False
    ):
        instance = super().from_game_dataset(game_dataset, concept)
        instance._unique_resample(strict=strict)
        return instance

    def _unique_resample(self, strict: bool = False):
        unique_boards: List[chess.Board] = []
        for game in self.games:
            board = chess.Board()
            for move in game.moves:
                board.push_san(move)
                if board not in unique_boards:
                    if strict:
                        for unique_board in unique_boards:
                            if (
                                board.move_stack[:8]
                                == unique_board.move_stack[:8]
                            ):  # Encodings can still be different
                                break
                        else:
                            unique_boards.append(board.copy())
                    else:
                        unique_boards.append(board.copy())
        self.unique_boards = unique_boards
