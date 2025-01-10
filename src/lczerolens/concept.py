"""Class for concept-based XAI methods."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from sklearn import metrics
from datasets import Features, Value, Sequence, ClassLabel

from lczerolens.board import LczeroBoard


class Concept(ABC):
    """
    Class for concept-based XAI methods.
    """

    @abstractmethod
    def compute_label(
        self,
        board: LczeroBoard,
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

    @property
    @abstractmethod
    def features(self) -> Features:
        """
        Return the features for the concept.
        """
        pass


class BinaryConcept(Concept):
    """
    Class for binary concept-based XAI methods.
    """

    features = Features(
        {
            "gameid": Value("string"),
            "moves": Sequence(Value("string")),
            "fen": Value("string"),
            "label": ClassLabel(num_classes=2),
        }
    )

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


class NullConcept(BinaryConcept):
    """
    Class for binary concept-based XAI methods.
    """

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> Any:
        """
        Compute the label for a given model and input.
        """
        return 0


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
        board: LczeroBoard,
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
        board: LczeroBoard,
    ) -> Any:
        """
        Compute the label for a given model and input.
        """
        return all(concept.compute_label(board) for concept in self.concepts)


class MulticlassConcept(Concept):
    """
    Class for multiclass concept-based XAI methods.
    """

    features = Features(
        {
            "gameid": Value("string"),
            "moves": Sequence(Value("string")),
            "fen": Value("string"),
            "label": Value("int32"),
        }
    )

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
            "precision": metrics.precision_score(labels, predictions, average="weighted"),
            "recall": metrics.recall_score(labels, predictions, average="weighted"),
            "f1": metrics.f1_score(labels, predictions, average="weighted"),
        }


class ContinuousConcept(Concept):
    """
    Class for continuous concept-based XAI methods.
    """

    features = Features(
        {
            "gameid": Value("string"),
            "moves": Sequence(Value("string")),
            "fen": Value("string"),
            "label": Value("float32"),
        }
    )

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


def concept_collate_fn(batch):
    boards = []
    labels = []
    for element in batch:
        board = LczeroBoard(element["fen"])
        for move in element["moves"]:
            board.push_san(move)
        boards.append(board)
        labels.append(element["label"])
    return boards, labels, batch


def concept_init_rel(output, infos):
    labels = infos[0]
    rel = torch.zeros_like(output)
    for i in range(rel.shape[0]):
        rel[i, labels[i]] = output[i, labels[i]]
    return rel
