"""Chess-facing views over canonical evaluator TensorDict batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from os import PathLike
from typing import Iterator, Sequence, overload

import chess
from tensordict import TensorDictBase

from lczerolens._codec import InputFormat, decode_move, encode_move
from lczerolens.provenance import ChessPlayer, EvaluationProvenance, PositionIdentity
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
class ActionEvaluationRecord:
    """Frozen evaluator preference for one legal UCI action."""

    move: str
    index: int
    logit: float
    probability: float
    rank: int

    def __post_init__(self) -> None:
        try:
            chess.Move.from_uci(self.move)
        except ValueError as error:
            raise ValueError("Recorded actions require a valid UCI move.") from error
        if isinstance(self.index, bool) or not isinstance(self.index, int) or not 0 <= self.index < 1858:
            raise ValueError("Recorded policy indices must be integers in [0, 1858).")
        if not math.isfinite(self.logit):
            raise ValueError("Recorded policy logits must be finite.")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ValueError("Recorded policy probabilities must be finite and in [0, 1].")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("Recorded policy ranks must be positive integers.")


@dataclass(frozen=True)
class ScalarEvaluationRecord:
    """Frozen scalar value with explicit origin and player perspective."""

    value: float
    origin: ValueOrigin
    perspective: ChessPlayer

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not -1 <= self.value <= 1:
            raise ValueError("Recorded scalar values must be finite and in [-1, 1].")
        if not isinstance(self.origin, ValueOrigin):
            raise ValueError("Recorded scalar origins must be ValueOrigin values.")
        if not isinstance(self.perspective, ChessPlayer):
            raise ValueError("Recorded scalar perspectives must be ChessPlayer values.")


@dataclass(frozen=True)
class WdlEvaluationRecord:
    """Frozen win/draw/loss probabilities for one absolute player."""

    win: float
    draw: float
    loss: float
    perspective: ChessPlayer

    def __post_init__(self) -> None:
        values = (self.win, self.draw, self.loss)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("Recorded WDL values must be finite probabilities in [0, 1].")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError("Recorded WDL probabilities must sum to one.")
        if not isinstance(self.perspective, ChessPlayer):
            raise ValueError("Recorded WDL perspectives must be ChessPlayer values.")


@dataclass(frozen=True)
class EvaluationRecord:
    """Immutable, tensor-free evidence produced by one evaluator call."""

    position: PositionIdentity
    provenance: EvaluationProvenance
    input_format: str
    policy: tuple[ActionEvaluationRecord, ...]
    wdl: WdlEvaluationRecord | None = None
    value: ScalarEvaluationRecord | None = None
    mlh: float | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.position, PositionIdentity):
            raise ValueError("Evaluation records require a PositionIdentity.")
        if not isinstance(self.provenance, EvaluationProvenance):
            raise ValueError("Evaluation records require EvaluationProvenance.")
        if not self.input_format:
            raise ValueError("Evaluation records require an input format.")
        try:
            InputFormat(self.input_format)
        except ValueError as error:
            raise ValueError(f"Unsupported evaluation input format {self.input_format!r}.") from error
        if any(not isinstance(action, ActionEvaluationRecord) for action in self.policy):
            raise ValueError("Recorded policy entries must be ActionEvaluationRecord values.")
        if self.wdl is not None and not isinstance(self.wdl, WdlEvaluationRecord):
            raise ValueError("Evaluation record WDL must be a WdlEvaluationRecord.")
        if self.value is not None and not isinstance(self.value, ScalarEvaluationRecord):
            raise ValueError("Evaluation record value must be a ScalarEvaluationRecord.")
        moves = [action.move for action in self.policy]
        indices = [action.index for action in self.policy]
        if len(moves) != len(set(moves)) or len(indices) != len(set(indices)):
            raise ValueError("Recorded policy actions and indices must be unique.")
        board = self.position.board()
        legal_moves = {move.uci() for move in board.legal_moves} if not board.is_game_over() else set()
        if set(moves) != legal_moves:
            raise ValueError("Recorded policy actions must exactly match the position's legal moves.")
        if moves != sorted(moves):
            raise ValueError("Recorded policy actions must use canonical UCI order.")
        if any(encode_move(board, chess.Move.from_uci(action.move)) != action.index for action in self.policy):
            raise ValueError("Recorded policy indices must match their moves in position context.")
        expected_ranks: dict[str, int] = {}
        previous_logit: float | None = None
        rank = 0
        for offset, action in enumerate(sorted(self.policy, key=lambda item: (-item.logit, item.move)), start=1):
            if previous_logit is None or action.logit != previous_logit:
                rank = offset
                previous_logit = action.logit
            expected_ranks[action.move] = rank
        if any(action.rank != expected_ranks[action.move] for action in self.policy):
            raise ValueError("Recorded policy ranks must agree with policy logits.")
        if self.policy and not math.isclose(sum(action.probability for action in self.policy), 1.0, abs_tol=1e-5):
            raise ValueError("Recorded legal policy probabilities must sum to one.")
        if self.wdl is not None and self.wdl.perspective is not self.position.player:
            raise ValueError("Recorded WDL perspective must match the position side to move.")
        if self.value is not None and self.value.perspective is not self.position.player:
            raise ValueError("Recorded scalar perspective must match the position side to move.")
        if self.mlh is not None and not math.isfinite(self.mlh):
            raise ValueError("Recorded MLH must be finite when present.")

    def to_bytes(self) -> bytes:
        """Return the canonical versioned JSON bytes for this record."""
        from lczerolens.serialization import serialize_evaluation_record

        return serialize_evaluation_record(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "EvaluationRecord":
        """Restore a record from canonical versioned JSON bytes."""
        from lczerolens.serialization import deserialize_evaluation_record

        return deserialize_evaluation_record(data)

    def digest(self) -> str:
        """Return the SHA-256 digest of the canonical bytes."""
        from lczerolens.serialization import evaluation_record_digest

        return evaluation_record_digest(self)

    def save(self, path: str | PathLike[str]) -> None:
        """Write the canonical bytes to ``path``."""
        from pathlib import Path

        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: str | PathLike[str]) -> "EvaluationRecord":
        """Load canonical bytes from ``path``."""
        from pathlib import Path

        return cls.from_bytes(Path(path).read_bytes())


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

    def __init__(
        self,
        board: chess.Board,
        tensors: TensorDictBase,
        provenance: EvaluationProvenance,
        input_format: str,
    ):
        if tensors.batch_size:
            raise ValueError("Evaluation requires one unbatched TensorDict row.")
        if not isinstance(provenance, EvaluationProvenance):
            raise TypeError("Evaluation requires EvaluationProvenance.")
        if not input_format:
            raise ValueError("Evaluation requires an input format.")
        self._position = board.copy(stack=True)
        self._tensors = tensors
        self._provenance = provenance
        self._input_format = input_format

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

    def record(self) -> EvaluationRecord:
        """Freeze the current runtime values into immutable tensor-free evidence."""
        player = ChessPlayer.from_color(self._position.turn)
        policy = tuple(
            ActionEvaluationRecord(
                move=action.move.uci(),
                index=action.index,
                logit=action.logit,
                probability=action.probability,
                rank=action.rank,
            )
            for action in self.policy.actions
        )
        wdl = self.wdl
        value = self.value
        return EvaluationRecord(
            position=PositionIdentity.from_board(self._position),
            provenance=self._provenance,
            input_format=self._input_format,
            policy=policy,
            wdl=(WdlEvaluationRecord(wdl.win, wdl.draw, wdl.loss, player) if wdl is not None else None),
            value=(ScalarEvaluationRecord(value.value, value.origin, player) if value is not None else None),
            mlh=self.mlh,
        )


class EvaluationBatch(Sequence[Evaluation]):
    """Batch-preserving collection of position evaluation views."""

    def __init__(
        self,
        boards: Sequence[chess.Board],
        tensors: TensorDictBase,
        provenance: EvaluationProvenance,
        input_format: str,
    ):
        if len(boards) != tensors.batch_size[0]:
            raise ValueError("Board count must match the TensorDict batch size.")
        if not isinstance(provenance, EvaluationProvenance):
            raise TypeError("EvaluationBatch requires EvaluationProvenance.")
        if not input_format:
            raise ValueError("EvaluationBatch requires an input format.")
        self._boards = tuple(board.copy(stack=True) for board in boards)
        self._tensors = tensors
        self._provenance = provenance
        self._input_format = input_format

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
            return EvaluationBatch(
                tuple(self._boards[position] for position in positions),
                self._tensors[index],
                self._provenance,
                self._input_format,
            )
        return Evaluation(self._boards[index], self._tensors[index], self._provenance, self._input_format)

    def __iter__(self) -> Iterator[Evaluation]:
        return (self[index] for index in range(len(self)))


__all__ = [
    "ActionEvaluation",
    "ActionEvaluationRecord",
    "Evaluation",
    "EvaluationBatch",
    "EvaluationRecord",
    "PolicyEvaluation",
    "ScalarEvaluation",
    "ScalarEvaluationRecord",
    "ValueOrigin",
    "WdlEvaluation",
    "WdlEvaluationRecord",
]
