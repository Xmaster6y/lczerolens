"""Class for concept-based XAI methods."""

import random
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple

import chess
import jsonlines
import torch
import tqdm
from sklearn import metrics

from lczerolens.encodings import board as board_encodings
from lczerolens.game.dataset import BoardDataset


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


class NullConcept(BinaryConcept):
    """
    Class for binary concept-based XAI methods.
    """

    def compute_label(
        self,
        board: chess.Board,
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
            "precision": metrics.precision_score(labels, predictions, average="weighted"),
            "recall": metrics.recall_score(labels, predictions, average="weighted"),
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


class ConceptDataset(BoardDataset):
    """
    Class for concept
    """

    def __init__(
        self,
        file_name: Optional[str] = None,
        boards: Optional[List[chess.Board]] = None,
        game_ids: Optional[List[str]] = None,
        concept: Optional[Concept] = None,
        labels: Optional[List[Any]] = None,
        first_n: Optional[int] = None,
    ):
        if file_name is None:
            super().__init__(file_name, boards, game_ids)
        else:
            self.boards = []
            self.game_ids = []
            self.labels = []
            with jsonlines.open(file_name) as reader:
                for obj in reader:
                    board = chess.Board(obj["fen"])
                    for move in obj["moves"]:
                        board.push_uci(move)

                    self.boards.append(board)
                    self.game_ids.append(obj["gameid"])
                    self.labels.append(obj["label"])
                    if first_n is not None and len(self.boards) >= first_n:
                        break
        self._concept = concept if concept is not None else NullConcept()
        if labels is not None:
            self.labels = labels
        elif not hasattr(self, "labels"):
            print("[INFO] Computing labels")
            self.labels = [
                self._concept.compute_label(board) for board in tqdm.tqdm(self.boards, bar_format="{l_bar}{bar}")
            ]

    def __getitem__(self, idx) -> Tuple[int, chess.Board, Any]:  # type: ignore
        board = self.boards[idx]
        label = self.labels[idx]
        return idx, board, label

    def save(self, file_name: str, n_history: int = 0, indices=None):
        print(f"[INFO] Saving boards to {file_name}")
        with jsonlines.open(file_name, "w") as writer:
            idx = 0
            for board, gameid, label in tqdm.tqdm(
                zip(self.boards, self.game_ids, self.labels),
                total=len(self.boards),
                bar_format="{l_bar}{bar}",
            ):
                if indices is not None and idx not in indices:
                    idx += 1
                    continue
                idx += 1
                working_board = board.copy(stack=n_history)

                writer.write(
                    {
                        "fen": working_board.root().fen(),
                        "moves": [move.uci() for move in working_board.move_stack],
                        "gameid": gameid,
                        "label": label,
                    }
                )

    @property
    def concept(self):
        return self._concept

    def set_concept(self, concept: Concept, **pbar_kwargs):
        self._concept = concept
        print("[INFO] Computing labels")
        self.labels = [
            self._concept.compute_label(board)
            for board in tqdm.tqdm(self.boards, bar_format="{l_bar}{bar}", **pbar_kwargs)
        ]

    @classmethod
    def from_board_dataset(cls, board_dataset: BoardDataset, concept: Concept, **pbar_kwargs):
        print("[INFO] Computing labels")
        labels = [
            concept.compute_label(board)
            for board in tqdm.tqdm(board_dataset.boards, bar_format="{l_bar}{bar}", **pbar_kwargs)
        ]
        return cls(
            boards=board_dataset.boards,
            game_ids=board_dataset.game_ids,
            concept=concept,
            labels=labels,
        )

    @staticmethod
    def collate_fn_tuple(batch):
        indices, boards, labels = zip(*batch)
        return tuple(indices), tuple(boards), tuple(labels)

    @staticmethod
    def collate_fn_tensor(batch):
        indices, boards, labels = zip(*batch)
        tensor_list = [board_encodings.board_to_input_tensor(board) for board in boards]
        batched_tensor = torch.stack(tensor_list, dim=0)
        return tuple(indices), batched_tensor, tuple(labels)

    def filter_(self, filter_fn: Callable):
        tuple_boards, tuple_labels, tuple_game_ids = zip(
            *[
                (board, label, game_id)
                for board, label, game_id in zip(self.boards, self.labels, self.game_ids)
                if filter_fn(board, label, game_id)
            ]
        )
        self.boards, self.labels, self.game_ids = (
            list(tuple_boards),
            list(tuple_labels),
            list(tuple_game_ids),
        )

    def filter_unique_(self):
        unique_boards: List[chess.Board] = []

        def filter_fn(board: chess.Board, label: Any, game_id: str):
            if board not in unique_boards:
                unique_boards.append(board)
                return True
            return False

        self.filter_(filter_fn)

    def filter_label_(self, filter_label: Any):
        if isinstance(self._concept, ContinuousConcept):
            raise ValueError("Continuous concept does not support resampling")
        self.filter_(lambda board, label, game_id: label == filter_label)

    def filter_even_(self, seed: int = 0):
        if isinstance(self._concept, ContinuousConcept):
            raise ValueError("Continuous concept does not support resampling")
        per_label_boards = {}
        for board, label in zip(self.boards, self.labels):
            if label not in per_label_boards:
                per_label_boards[label] = [board]
            else:
                per_label_boards[label].append(board)

        min_len = min(len(boards) for boards in per_label_boards.values())
        self.boards = []
        self.labels = []
        random.seed(seed)
        for label, label_boards in per_label_boards.items():
            self.boards.extend(random.sample(label_boards, min_len))
            self.labels.extend([label] * min_len)
