"""Class for concept-based XAI methods.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import chess
import jsonlines
import torch
from sklearn import metrics
from torch.utils.data import Dataset

from lczerolens.game.dataset import BoardDataset, GameDataset
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


class ConceptDataset(Dataset):
    """
    Class for concept
    """

    def __init__(
        self,
        concept: Concept,
        file_name: Optional[str] = None,
        boards: Optional[List[chess.Board]] = None,
        labels: Optional[List[Any]] = None,
    ):
        if boards is not None and file_name is not None:
            raise ValueError("Either boards or file_name must be provided")
        elif boards is not None:
            self.boards = boards
        else:
            self.boards = []
            if file_name is not None:
                with jsonlines.open(file_name) as reader:
                    for obj in reader:
                        board = chess.Board(obj["fen"])
                        for move in obj["moves"]:
                            board.push_san(move)

                        self.boards.append(board)
        self._concept = concept
        if labels is not None:
            self.labels = labels
        else:
            self.labels = [
                self._concept.compute_label(board) for board in self.boards
            ]
        self.tensors = [
            board_utils.board_to_tensor112x8x8(board) for board in self.boards
        ]

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx) -> chess.Board:
        tensor = self.tensors[idx]
        board = self.boards[idx]
        label = self.labels[idx]
        return tensor, board, label

    @property
    def concept(self):
        return self._concept

    @concept.setter
    def concept(self, concept):
        self._concept = concept
        self.labels = [
            self._concept.compute_label(board) for board in self.boards
        ]

    @classmethod
    def from_game_dataset(
        cls, game_dataset: GameDataset, concept: Concept, n_history: int = 0
    ):
        board_dataset = BoardDataset.from_game_dataset(
            game_dataset=game_dataset, n_history=n_history
        )
        return cls.from_board_dataset(board_dataset, concept)

    @classmethod
    def from_board_dataset(cls, board_dataset: BoardDataset, concept: Concept):
        labels = [
            concept.compute_label(board) for board in board_dataset.boards
        ]
        return cls(concept, boards=board_dataset.boards, labels=labels)

    @staticmethod
    def collate_fn(batch):
        tensor_list, _, label_list = zip(*batch)
        batched_tensor = torch.stack(tensor_list, dim=0)
        return batched_tensor, label_list


class UniqueConceptDataset(ConceptDataset):
    """
    Class for unique board concept
    """

    def __init__(
        self,
        concept: Concept,
        file_name: Optional[str] = None,
        boards: Optional[List[chess.Board]] = None,
        labels: Optional[List[Any]] = None,
    ):
        super().__init__(concept, file_name, boards, labels)
        self._unique_resample()

    @classmethod
    def from_game_dataset(
        cls, game_dataset: GameDataset, concept: Concept, n_history: int = 0
    ):
        instance = super().from_game_dataset(game_dataset, concept, n_history)
        instance._unique_resample()
        return instance

    @classmethod
    def from_board_dataset(
        cls,
        board_dataset: BoardDataset,
        concept: Concept,
    ):
        instance = super().from_board_dataset(board_dataset, concept)
        instance._unique_resample()
        return instance

    @classmethod
    def from_concept_dataset(cls, concept_dataset: ConceptDataset):
        return cls(
            concept_dataset.concept,
            boards=concept_dataset.boards,
            labels=concept_dataset.labels,
        )

    def _unique_resample(self):
        unique_boards: List[chess.Board] = []
        labels = []
        for board, label in zip(self.boards, self.labels):
            if board not in unique_boards:
                unique_boards.append(board)
                labels.append(label)
        self.boards = unique_boards
        self.labels = labels
