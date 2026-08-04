"""Chess-facing views over canonical evaluator TensorDict batches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence, overload

import chess
from tensordict import TensorDictBase

from lczerolens._codec import decode_move
from lczerolens.schema import LczeroKeys


class ValueOrigin(str, Enum):
    """Origin of a standardized scalar evaluation."""

    NATIVE = "native"
    DERIVED_FROM_WDL = "derived_from_wdl"


@dataclass(frozen=True)
class ScalarEvaluation:
    """One scalar value and whether the network emitted or derived it."""

    value: float
    origin: ValueOrigin
    perspective: chess.Color


@dataclass(frozen=True)
class WdlEvaluation:
    """Win, draw, and loss probabilities from the side-to-move perspective."""

    win: float
    draw: float
    loss: float
    perspective: chess.Color


@dataclass(frozen=True)
class ActionEvaluation:
    """Evaluator preference for one legal action."""

    move: chess.Move
    index: int
    logit: float
    probability: float
    rank: int


class PolicyEvaluation:
    """Legal-move view over one raw and standardized policy row."""

    def __init__(self, board: chess.Board, tensors: TensorDictBase):
        self._board = board.copy(stack=True)
        self._tensors = tensors
        mask = tensors[LczeroKeys.INPUT_LEGAL_MASK]
        logits = tensors[LczeroKeys.NETWORK_POLICY_LOGITS]
        probabilities = tensors[LczeroKeys.EVALUATION_POLICY]
        indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
        ranked = sorted(indices, key=lambda index: (-float(logits[index].item()), decode_move(board, index).uci()))
        ranks: dict[int, int] = {}
        previous_logit: float | None = None
        rank = 0
        for offset, index in enumerate(ranked, start=1):
            logit = float(logits[index].item())
            if previous_logit is None or logit != previous_logit:
                rank = offset
                previous_logit = logit
            ranks[index] = rank
        self._actions = tuple(
            ActionEvaluation(
                move=decode_move(board, index),
                index=index,
                logit=float(logits[index].item()),
                probability=float(probabilities[index].item()),
                rank=ranks[index],
            )
            for index in sorted(indices, key=lambda item: decode_move(board, item).uci())
        )
        self._by_uci = {action.move.uci(): action for action in self._actions}

    @property
    def actions(self) -> tuple[ActionEvaluation, ...]:
        """Legal actions in stable UCI order."""
        return self._actions

    @property
    def best_move(self) -> chess.Move | None:
        """Maximum-logit legal move with a stable UCI tie-break."""
        return min(
            (action.move for action in self._actions if action.rank == 1),
            key=lambda move: move.uci(),
            default=None,
        )

    @property
    def is_defined(self) -> bool:
        """Whether the position has a semantic legal policy."""
        return bool(self._actions)

    def __getitem__(self, move: str | chess.Move) -> ActionEvaluation:
        uci = move if isinstance(move, str) else move.uci()
        try:
            return self._by_uci[uci]
        except KeyError as error:
            raise KeyError(f"{uci!r} is not a legal evaluated move for this position.") from error

    def top(self, count: int) -> tuple[ActionEvaluation, ...]:
        """Return the ``count`` highest-ranked legal actions."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer.")
        return tuple(sorted(self._actions, key=lambda action: (action.rank, action.move.uci()))[:count])


class Evaluation:
    """One position bound to one row of canonical evaluator tensors."""

    def __init__(self, board: chess.Board, tensors: TensorDictBase):
        if tensors.batch_size:
            raise ValueError("Evaluation requires one unbatched TensorDict row.")
        self._position = board.copy(stack=True)
        self._tensors = tensors

    @property
    def position(self) -> chess.Board:
        """A defensive full-history copy of the evaluated position."""
        return self._position.copy(stack=True)

    @property
    def tensors(self) -> TensorDictBase:
        """The unbatched TensorDict row underlying this view."""
        return self._tensors

    @property
    def policy(self) -> PolicyEvaluation:
        """Legal policy view over the current evaluator tensors."""
        return PolicyEvaluation(self._position, self._tensors)

    @property
    def wdl(self) -> WdlEvaluation | None:
        if LczeroKeys.NETWORK_WDL not in self._tensors.keys(include_nested=True, leaves_only=True):
            return None
        values = self._tensors[LczeroKeys.NETWORK_WDL].reshape(-1).tolist()
        return WdlEvaluation(*(float(value) for value in values), perspective=self._position.turn)

    @property
    def value(self) -> ScalarEvaluation | None:
        if LczeroKeys.EVALUATION_VALUE not in self._tensors.keys(include_nested=True, leaves_only=True):
            return None
        origin = (
            ValueOrigin.NATIVE
            if LczeroKeys.NETWORK_VALUE in self._tensors.keys(include_nested=True, leaves_only=True)
            else ValueOrigin.DERIVED_FROM_WDL
        )
        return ScalarEvaluation(
            float(self._tensors[LczeroKeys.EVALUATION_VALUE].reshape(-1)[0]),
            origin,
            self._position.turn,
        )

    @property
    def mlh(self) -> float | None:
        if LczeroKeys.NETWORK_MLH not in self._tensors.keys(include_nested=True, leaves_only=True):
            return None
        return float(self._tensors[LczeroKeys.NETWORK_MLH].reshape(-1)[0])


class EvaluationBatch(Sequence[Evaluation]):
    """Batch-preserving collection of position evaluation views."""

    def __init__(self, boards: Sequence[chess.Board], tensors: TensorDictBase):
        if len(boards) != tensors.batch_size[0]:
            raise ValueError("Board count must match the TensorDict batch size.")
        self._boards = tuple(board.copy(stack=True) for board in boards)
        self._tensors = tensors

    @property
    def tensors(self) -> TensorDictBase:
        return self._tensors

    def __len__(self) -> int:
        return len(self._boards)

    @overload
    def __getitem__(self, index: int) -> Evaluation: ...

    @overload
    def __getitem__(self, index: slice) -> "EvaluationBatch": ...

    def __getitem__(self, index: int | slice) -> Evaluation | "EvaluationBatch":
        if isinstance(index, slice):
            positions = range(*index.indices(len(self)))
            return EvaluationBatch(tuple(self._boards[position] for position in positions), self._tensors[index])
        return Evaluation(self._boards[index], self._tensors[index])

    def __iter__(self) -> Iterator[Evaluation]:
        return (self[index] for index in range(len(self)))


__all__ = [
    "ActionEvaluation",
    "Evaluation",
    "EvaluationBatch",
    "PolicyEvaluation",
    "ScalarEvaluation",
    "ValueOrigin",
    "WdlEvaluation",
]
