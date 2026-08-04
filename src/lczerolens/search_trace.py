"""Engine-independent records for search evidence.

The records in this module describe what a search source actually exposed.  A
trace's :class:`SearchCapability` is a promise to consumers; absent engine data
must stay ``None`` rather than being reconstructed or guessed by an adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import chess


ParameterValue: TypeAlias = str | int | float | bool | None


class SearchCapability(str, Enum):
    """Ordered information levels supplied by a search trace."""

    ROOT_RESULT = "root_result"
    ROOT_ACTION_STATS = "root_action_stats"
    ROOT_SNAPSHOTS = "root_snapshots"
    FULL_EVENTS = "full_events"
    REPLAYABLE = "replayable"

    @property
    def level(self) -> int:
        """Return the information ordering used by capability checks."""
        return _CAPABILITY_LEVEL[self]


_CAPABILITY_LEVEL = {
    SearchCapability.ROOT_RESULT: 0,
    SearchCapability.ROOT_ACTION_STATS: 1,
    SearchCapability.ROOT_SNAPSHOTS: 2,
    SearchCapability.FULL_EVENTS: 3,
    SearchCapability.REPLAYABLE: 4,
}


class SearchCapabilityError(RuntimeError):
    """Raised when a consumer asks a trace for evidence it does not contain."""


class ValuePerspective(str, Enum):
    """Player whose expected outcome a scalar value or WDL describes."""

    SIDE_TO_MOVE = "side_to_move"
    ROOT_PLAYER = "root_player"
    WHITE = "white"
    BLACK = "black"


class ChessPlayer(str, Enum):
    """An absolute chess player used to identify the root side to move."""

    WHITE = "white"
    BLACK = "black"


class SearchBudgetUnit(str, Enum):
    """Unit used for a requested or observed search budget."""

    NODES = "nodes"
    VISITS = "visits"
    SIMULATIONS = "simulations"
    TIME_MS = "time_ms"
    DEPTH = "depth"


_COUNT_BUDGET_UNITS = {
    SearchBudgetUnit.NODES,
    SearchBudgetUnit.VISITS,
    SearchBudgetUnit.SIMULATIONS,
    SearchBudgetUnit.DEPTH,
}


@dataclass(frozen=True)
class SearchParameter:
    """One source-specific search option retained as provenance."""

    name: str
    value: ParameterValue

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Search parameter names must not be empty.")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError(f"Search parameter {self.name!r} must be finite.")


@dataclass(frozen=True)
class SearchProvenance:
    """Identity and configuration of the producer of a trace."""

    source: str
    engine: str | None = None
    engine_version: str | None = None
    network: str | None = None
    network_checksum: str | None = None
    parameters: tuple[SearchParameter, ...] = ()

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Search provenance source must not be empty.")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Search provenance parameter names must be unique.")


@dataclass(frozen=True)
class SearchBudget:
    """Requested and observed work for one root result or snapshot."""

    unit: SearchBudgetUnit
    requested: float | int | None = None
    observed: float | int | None = None

    def __post_init__(self) -> None:
        if self.requested is None and self.observed is None:
            raise ValueError("A search budget needs a requested or observed value.")
        for name, value in (("requested", self.requested), ("observed", self.observed)):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Search budget {name} must be finite and non-negative.")
            if value is not None and self.unit in _COUNT_BUDGET_UNITS and not isinstance(value, int):
                raise ValueError(f"Search budget {name} must be an integer for {self.unit.value}.")


@dataclass(frozen=True)
class Wdl:
    """Win/draw/loss probabilities from one explicit player perspective."""

    win: float
    draw: float
    loss: float
    perspective: ValuePerspective

    def __post_init__(self) -> None:
        probabilities = (self.win, self.draw, self.loss)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in probabilities):
            raise ValueError("WDL entries must be finite probabilities in [0, 1].")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("WDL probabilities must sum to one.")

    def scalar(self, draw_score: float = 0.0) -> float:
        """Convert WDL to ``win + draw_score * draw - loss``."""
        if not math.isfinite(draw_score):
            raise ValueError("draw_score must be finite.")
        return self.win + draw_score * self.draw - self.loss

    @classmethod
    def from_win_loss_draw(
        cls,
        win_minus_loss: float,
        draw: float,
        perspective: ValuePerspective,
    ) -> Wdl:
        """Build WDL from lc0-style ``WL = win - loss`` and draw probability."""
        win = (1.0 - draw + win_minus_loss) / 2.0
        loss = (1.0 - draw - win_minus_loss) / 2.0
        return cls(win=win, draw=draw, loss=loss, perspective=perspective)


@dataclass(frozen=True)
class PositionEvaluation:
    """An optional scalar and/or WDL evaluation of a position."""

    perspective: ValuePerspective
    value: float | None = None
    wdl: Wdl | None = None

    def __post_init__(self) -> None:
        if self.value is None and self.wdl is None:
            raise ValueError("A position evaluation needs a scalar value or WDL.")
        if self.value is not None and (not math.isfinite(self.value) or not -1 <= self.value <= 1):
            raise ValueError("Scalar position values must be finite and in [-1, 1].")
        if self.wdl is not None and self.wdl.perspective is not self.perspective:
            raise ValueError("Scalar and WDL perspectives must agree.")


@dataclass(frozen=True)
class PrincipalVariation:
    """A source-reported sequence of UCI moves, without inferred continuations."""

    moves: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.moves or any(not move for move in self.moves):
            raise ValueError("A principal variation must contain non-empty UCI moves.")


@dataclass(frozen=True)
class EdgeStatistics:
    """Statistics for one directed edge, expressed in its stated perspective.

    ``prior`` is P, ``visits`` is N, ``total_value`` is W, ``mean_value`` is
    Q, and ``exploration`` is the source-reported U term.  U is retained rather
    than recomputed because its formula is implementation-specific.
    """

    move: str
    perspective: ValuePerspective
    prior: float | None = None
    visits: int | None = None
    total_value: float | None = None
    mean_value: float | None = None
    exploration: float | None = None

    def __post_init__(self) -> None:
        if not self.move:
            raise ValueError("Edge moves must be non-empty UCI strings.")
        if self.prior is not None and (not math.isfinite(self.prior) or not 0 <= self.prior <= 1):
            raise ValueError("P must be a finite probability in [0, 1].")
        if self.visits is not None and (
            not isinstance(self.visits, int) or isinstance(self.visits, bool) or self.visits < 0
        ):
            raise ValueError("N must be a non-negative integer.")
        for name, value in (("W", self.total_value), ("Q", self.mean_value), ("U", self.exploration)):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when present.")
        if self.mean_value is not None and not -1 <= self.mean_value <= 1:
            raise ValueError("Q must be in [-1, 1].")
        if self.exploration is not None and self.exploration < 0:
            raise ValueError("U must be non-negative.")
        if self.visits is not None and self.total_value is not None and abs(self.total_value) > self.visits + 1e-6:
            raise ValueError("W must be in [-N, N] for values bounded to [-1, 1].")
        if self.visits is not None and self.total_value is not None and self.mean_value is not None:
            expected = self.total_value / self.visits if self.visits else 0.0
            if not math.isclose(self.mean_value, expected, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("Q must equal W / N, with Q = 0 when N = 0.")


@dataclass(frozen=True)
class RootAction:
    """One legal root action and exactly the statistics the source exposed."""

    statistics: EdgeStatistics
    evaluation: PositionEvaluation | None = None
    leaf_evaluation: PositionEvaluation | None = None
    principal_variation: PrincipalVariation | None = None

    def __post_init__(self) -> None:
        if self.principal_variation is not None and self.principal_variation.moves[0] != self.statistics.move:
            raise ValueError("A root action's principal variation must start with that action.")


@dataclass(frozen=True)
class RootSelection:
    """Selected root move and the rule used to resolve sampling or ties."""

    move: str
    rule: str
    tie_break: str
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not self.move or not self.rule or not self.tie_break:
            raise ValueError("Root selection move, rule, and tie-break must be explicit.")
        if self.temperature is not None and (not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError("Root selection temperature must be finite and non-negative.")


@dataclass(frozen=True)
class RootSnapshot:
    """Root evidence at one point in a search.

    ``actions=None`` means the producer did not expose action statistics;
    ``actions=()`` means it explicitly exposed an empty legal-action set.
    """

    sequence: int
    selection: RootSelection | None = None
    evaluation: PositionEvaluation | None = None
    budget: SearchBudget | None = None
    actions: tuple[RootAction, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("Root snapshot sequence must be non-negative.")
        if self.selection is None and self.evaluation is None and self.actions is None:
            raise ValueError("A root snapshot needs a selection, evaluation, or exposed actions.")
        if self.actions is not None:
            moves = [action.statistics.move for action in self.actions]
            if len(moves) != len(set(moves)):
                raise ValueError("Root snapshot actions must have unique moves.")
            if self.selection is not None and self.selection.move not in moves:
                raise ValueError("The selected root move must appear in the exposed actions.")
            priors = [action.statistics.prior for action in self.actions]
            if (
                priors
                and all(prior is not None for prior in priors)
                and not math.isclose(
                    sum(prior for prior in priors if prior is not None), 1.0, rel_tol=1e-6, abs_tol=1e-6
                )
            ):
                raise ValueError("Exposed root priors must sum to one.")


@dataclass(frozen=True)
class PathStep:
    """One selected parent-edge-child transition in a simulation."""

    node_id: str
    move: str
    child_id: str


@dataclass(frozen=True)
class EvaluatorCall:
    """Evaluator metadata retained for an expanded or evaluated leaf."""

    dtype: str
    source_device: str
    search_device: str
    legal_policy_logits: tuple[tuple[str, float], ...] | None = None

    def __post_init__(self) -> None:
        if not self.dtype or not self.source_device or not self.search_device:
            raise ValueError("Evaluator dtype and device fields must be explicit.")
        if self.legal_policy_logits is not None:
            moves = [move for move, _ in self.legal_policy_logits]
            values = [value for _, value in self.legal_policy_logits]
            if len(moves) != len(set(moves)) or any(not move for move in moves):
                raise ValueError("Legal policy logits must have unique, non-empty moves.")
            if any(not math.isfinite(value) for value in values):
                raise ValueError("Legal policy logits must be finite.")


@dataclass(frozen=True)
class LeafRecord:
    """Leaf reached by a simulation and the value returned for backup."""

    node_id: str
    evaluation: PositionEvaluation
    terminal: bool
    evaluator: EvaluatorCall | None = None


@dataclass(frozen=True)
class NodeExpansion:
    """Legal priors created when a node was expanded."""

    node_id: str
    edges: tuple[EdgeStatistics, ...]

    def __post_init__(self) -> None:
        moves = [edge.move for edge in self.edges]
        if len(moves) != len(set(moves)):
            raise ValueError("Expansion edges must have unique moves.")
        if any(edge.prior is None for edge in self.edges):
            raise ValueError("Every expanded edge must have an explicit prior.")
        if self.edges and not math.isclose(
            sum(edge.prior for edge in self.edges if edge.prior is not None), 1.0, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("Expanded legal priors must sum to one.")


@dataclass(frozen=True)
class BackupUpdate:
    """Pre/post state for one backed-up edge."""

    node_id: str
    signed_value: float
    before: EdgeStatistics
    after: EdgeStatistics

    def __post_init__(self) -> None:
        if not math.isfinite(self.signed_value) or not -1 <= self.signed_value <= 1:
            raise ValueError("Backed-up values must be finite and in [-1, 1].")
        if self.before.move != self.after.move or self.before.perspective is not self.after.perspective:
            raise ValueError("Backup pre/post records must describe the same edge and perspective.")
        before = (self.before.visits, self.before.total_value, self.before.mean_value)
        after = (self.after.visits, self.after.total_value, self.after.mean_value)
        if any(value is None for value in before + after):
            raise ValueError("Backup updates require pre/post N, W, and Q.")
        if self.after.visits != self.before.visits + 1:
            raise ValueError("Backup updates require N_post = N_pre + 1.")
        if not math.isclose(
            self.after.total_value, self.before.total_value + self.signed_value, rel_tol=1e-6, abs_tol=1e-6
        ):
            raise ValueError("Backup updates require W_post = W_pre + signed_value.")


@dataclass(frozen=True)
class SimulationEvent:
    """Append-only evidence for one reference-search simulation."""

    event_id: str
    simulation: int
    path: tuple[PathStep, ...]
    leaf: LeafRecord
    backups: tuple[BackupUpdate, ...]
    expansion: NodeExpansion | None = None
    root_before: tuple[EdgeStatistics, ...] | None = None
    root_after: tuple[EdgeStatistics, ...] | None = None

    @property
    def replayable(self) -> bool:
        """Whether this event contains the state transitions needed for replay."""
        if not self.root_before or not self.root_after:
            return False
        before = {edge.move: edge for edge in self.root_before}
        after = {edge.move: edge for edge in self.root_after}
        if len(before) != len(self.root_before) or len(after) != len(self.root_after) or before.keys() != after.keys():
            return False
        changed = [(before[move], after[move]) for move in before if before[move] != after[move]]
        return len(changed) == 1 and any(
            update.before == changed[0][0] and update.after == changed[0][1] for update in self.backups
        )

    def __post_init__(self) -> None:
        if (
            not self.event_id
            or not isinstance(self.simulation, int)
            or isinstance(self.simulation, bool)
            or self.simulation < 0
        ):
            raise ValueError("Simulation events need an ID and non-negative index.")


@dataclass(frozen=True)
class SearchTrace:
    """A capability-labelled sequence of root snapshots and optional events."""

    root_fen: str
    root_player: ChessPlayer
    capability: SearchCapability
    provenance: SearchProvenance
    snapshots: tuple[RootSnapshot, ...]
    events: tuple[SimulationEvent, ...] | None = None
    root_expansion: NodeExpansion | None = None
    root_evaluator: EvaluatorCall | None = None
    root_start_fen: str | None = None
    root_move_history: tuple[str, ...] | None = None
    schema_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.root_player, ChessPlayer):
            raise ValueError("root_player must be ChessPlayer.WHITE or ChessPlayer.BLACK.")
        try:
            board = chess.Board(self.root_fen)
        except ValueError as error:
            raise ValueError("root_fen must be a valid chess FEN.") from error
        fen_player = ChessPlayer.WHITE if board.turn is chess.WHITE else ChessPlayer.BLACK
        if self.root_player is not fen_player:
            raise ValueError("root_player must match the side to move in root_fen.")
        if (self.root_start_fen is None) != (self.root_move_history is None):
            raise ValueError("root history requires both a starting FEN and move sequence.")
        if self.root_start_fen is not None and self.root_move_history is not None:
            try:
                history_board = chess.Board(self.root_start_fen)
                for move_uci in self.root_move_history:
                    move = chess.Move.from_uci(move_uci)
                    if move not in history_board.legal_moves:
                        raise ValueError("Root history moves must be legal from root_start_fen.")
                    history_board.push(move)
            except ValueError as error:
                raise ValueError("root history must be a legal sequence from root_start_fen.") from error
            if history_board.fen() != self.root_fen:
                raise ValueError("root history must reconstruct root_fen.")
        legal_moves = {move.uci() for move in board.legal_moves}
        for snapshot in self.snapshots:
            if snapshot.selection is not None and snapshot.selection.move not in legal_moves:
                raise ValueError("The selected root move must be legal in root_fen.")
            for action in snapshot.actions or ():
                if action.statistics.move not in legal_moves:
                    raise ValueError("Every root action must be legal in root_fen.")
                if action.principal_variation is not None:
                    variation_board = board.copy(stack=False)
                    for move_uci in action.principal_variation.moves:
                        try:
                            move = chess.Move.from_uci(move_uci)
                        except ValueError as error:
                            raise ValueError("Principal variations must contain valid UCI moves.") from error
                        if move not in variation_board.legal_moves:
                            raise ValueError("Principal variation moves must be legal in sequence from root_fen.")
                        variation_board.push(move)
        if not self.snapshots:
            raise ValueError("A search trace must contain at least one root snapshot.")
        if not any(snapshot.selection is not None or snapshot.evaluation is not None for snapshot in self.snapshots):
            raise ValueError("A search trace must contain a root selection or evaluation.")
        sequences = [snapshot.sequence for snapshot in self.snapshots]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise ValueError("Root snapshot sequence numbers must be unique and increasing.")
        if self.supports(SearchCapability.ROOT_ACTION_STATS) and any(
            snapshot.actions is None for snapshot in self.snapshots
        ):
            raise ValueError("Root-action capability requires actions in every snapshot.")
        if self.supports(SearchCapability.ROOT_SNAPSHOTS) and any(
            snapshot.budget is None for snapshot in self.snapshots
        ):
            raise ValueError("Root-snapshot capability requires a budget on every snapshot.")
        if self.supports(SearchCapability.FULL_EVENTS) and self.events is None:
            raise ValueError("Full-event capability requires an events collection.")
        if self.events is not None:
            event_ids = [event.event_id for event in self.events]
            simulations = [event.simulation for event in self.events]
            if len(event_ids) != len(set(event_ids)):
                raise ValueError("Simulation event IDs must be unique.")
            if simulations != sorted(simulations) or len(simulations) != len(set(simulations)):
                raise ValueError("Simulation indices must be unique and increasing.")
        if self.capability is SearchCapability.REPLAYABLE and any(not event.replayable for event in self.events or ()):
            raise ValueError("Replayable capability requires replay state on every event.")
        if self.capability is SearchCapability.REPLAYABLE and self.events:
            for previous, current in zip(self.events, self.events[1:]):
                if previous.root_after != current.root_before:
                    raise ValueError("Replayable root states must chain between simulation events.")
            final_actions = self.snapshots[-1].actions
            final_root_state = self.events[-1].root_after
            final_action_stats = {action.statistics.move: action.statistics for action in final_actions or ()}
            final_event_stats = {edge.move: edge for edge in final_root_state or ()}
            if final_actions is None or final_root_state is None or final_action_stats != final_event_stats:
                raise ValueError("Replayable root state must agree with the final root snapshot.")

    def supports(self, capability: SearchCapability) -> bool:
        """Return whether this trace advertises at least ``capability``."""
        return self.capability.level >= capability.level

    def require(self, capability: SearchCapability) -> SearchTrace:
        """Return this trace or reject an unsupported evidence claim."""
        if not self.supports(capability):
            raise SearchCapabilityError(
                f"Trace capability {self.capability.value!r} does not support {capability.value!r}."
            )
        return self


__all__ = [
    "BackupUpdate",
    "ChessPlayer",
    "EdgeStatistics",
    "EvaluatorCall",
    "LeafRecord",
    "NodeExpansion",
    "ParameterValue",
    "PathStep",
    "PositionEvaluation",
    "PrincipalVariation",
    "RootAction",
    "RootSelection",
    "RootSnapshot",
    "SearchBudget",
    "SearchBudgetUnit",
    "SearchCapability",
    "SearchCapabilityError",
    "SearchParameter",
    "SearchProvenance",
    "SearchTrace",
    "SimulationEvent",
    "ValuePerspective",
    "Wdl",
]
