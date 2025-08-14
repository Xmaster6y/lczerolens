"""Class for concept-based XAI methods."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

import torch
import chess

from lczerolens.board import LczeroBoard
from lczerolens.model import LczeroModel, PolicyFlow


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

    @staticmethod
    @abstractmethod
    def get_dataset_feature():
        """Returns the feature for the dataset."""
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
        try:
            from sklearn import metrics
        except ImportError as e:
            raise ImportError("scikit-learn is required to compute metrics.") from e
        return {
            "accuracy": metrics.accuracy_score(labels, predictions),
            "precision": metrics.precision_score(labels, predictions),
            "recall": metrics.recall_score(labels, predictions),
            "f1": metrics.f1_score(labels, predictions),
        }

    @staticmethod
    def get_dataset_feature():
        """Returns the feature for the dataset."""
        try:
            from datasets import ClassLabel
        except ImportError as e:
            raise ImportError("datasets is required to get the dataset features.") from e
        return ClassLabel(num_classes=2)


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

    @staticmethod
    def compute_metrics(
        predictions,
        labels,
    ):
        """
        Compute the metrics for a given model and input.
        """
        try:
            from sklearn import metrics
        except ImportError as e:
            raise ImportError("scikit-learn is required to compute metrics.") from e
        return {
            "accuracy": metrics.accuracy_score(labels, predictions),
            "precision": metrics.precision_score(labels, predictions, average="weighted"),
            "recall": metrics.recall_score(labels, predictions, average="weighted"),
            "f1": metrics.f1_score(labels, predictions, average="weighted"),
        }

    @staticmethod
    def get_dataset_feature():
        """Returns the feature for the dataset."""
        try:
            from datasets import Value
        except ImportError as e:
            raise ImportError("datasets is required to get the dataset features.") from e
        return (Value("int32"),)


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
        try:
            from sklearn import metrics
        except ImportError as e:
            raise ImportError("scikit-learn is required to compute metrics.") from e
        return {
            "rmse": metrics.root_mean_squared_error(labels, predictions),
            "mae": metrics.mean_absolute_error(labels, predictions),
            "r2": metrics.r2_score(labels, predictions),
        }

    @staticmethod
    def get_dataset_feature():
        """Returns the feature for the dataset."""
        try:
            from datasets import Value
        except ImportError as e:
            raise ImportError("datasets is required to get the dataset features.") from e
        return Value("float32")


class HasPiece(BinaryConcept):
    """Class for material concept-based XAI methods."""

    def __init__(
        self,
        piece: str,
        relative: bool = True,
    ):
        """Initialize the class."""
        self.piece = chess.Piece.from_symbol(piece)
        self.relative = relative

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """Compute the label for a given model and input."""
        if self.relative:
            color = self.piece.color if board.turn else not self.piece.color
        else:
            color = self.piece.color
        squares = board.pieces(self.piece.piece_type, color)
        return 1 if len(squares) > 0 else 0


# Material concepts


class HasMaterialAdvantage(BinaryConcept):
    """Class for material concept-based XAI methods.

    Attributes
    ----------
    piece_values : Dict[int, int]
        The piece values.
    """

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(
        self,
        relative: bool = True,
    ):
        """
        Initialize the class.
        """
        self.relative = relative

    def compute_label(
        self,
        board: LczeroBoard,
        piece_values: Optional[Dict[int, int]] = None,
    ) -> int:
        """
        Compute the label for a given model and input.
        """
        if piece_values is None:
            piece_values = self.piece_values
        if self.relative:
            us, them = board.turn, not board.turn
        else:
            us, them = chess.WHITE, chess.BLACK
        our_value = 0
        their_value = 0
        for piece in range(1, 7):
            our_value += len(board.pieces(piece, us)) * piece_values[piece]
            their_value += len(board.pieces(piece, them)) * piece_values[piece]
        return 1 if our_value > their_value else 0


# Move concepts


class BestLegalMove(MulticlassConcept):
    """Class for move concept-based XAI methods."""

    def __init__(
        self,
        model: LczeroModel,
    ):
        """Initialize the class."""
        self.policy_flow = PolicyFlow(model)

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """Compute the label for a given model and input."""
        (policy,) = self.policy_flow(board)
        policy = torch.softmax(policy.squeeze(0), dim=-1)

        legal_move_indices = [LczeroBoard.encode_move(move, board.turn) for move in board.legal_moves]
        sub_index = policy[legal_move_indices].argmax().item()
        return legal_move_indices[sub_index]


class PieceBestLegalMove(BinaryConcept):
    """Class for move concept-based XAI methods."""

    def __init__(
        self,
        model: LczeroModel,
        piece: str,
    ):
        """Initialize the class."""
        self.policy_flow = PolicyFlow(model)
        self.piece = chess.Piece.from_symbol(piece)

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """Compute the label for a given model and input."""
        (policy,) = self.policy_flow(board)
        policy = torch.softmax(policy.squeeze(0), dim=-1)

        legal_moves = list(board.legal_moves)
        legal_move_indices = [LczeroBoard.encode_move(move, board.turn) for move in legal_moves]
        sub_index = policy[legal_move_indices].argmax().item()
        best_legal_move = legal_moves[sub_index]
        if board.piece_at(best_legal_move.from_square) == self.piece:
            return 1
        return 0


# Threat concepts


class HasThreat(BinaryConcept):
    """
    Class for material concept-based XAI methods.
    """

    def __init__(
        self,
        piece: str,
        relative: bool = True,
    ):
        """
        Initialize the class.
        """
        self.piece = chess.Piece.from_symbol(piece)
        self.relative = relative

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """
        Compute the label for a given model and input.
        """
        if self.relative:
            color = self.piece.color if board.turn else not self.piece.color
        else:
            color = self.piece.color
        squares = board.pieces(self.piece.piece_type, color)
        for square in squares:
            if board.is_attacked_by(not color, square):
                return 1
        return 0


class HasMateThreat(BinaryConcept):
    """
    Class for material concept-based XAI methods.
    """

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """
        Compute the label for a given model and input.
        """
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return 1
            board.pop()
        return 0
