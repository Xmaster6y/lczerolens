"""A small deterministic neural-MCTS implementation for auditable experiments.

This module deliberately does not model lc0's batching, virtual visits, tree
reuse, transpositions, pruning, or FPU behaviour.  It emits the public
``SearchTrace`` schema after every sequential simulation instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from numbers import Real
from typing import Callable, Protocol

import chess
import torch
from tensordict import TensorDict

from lczerolens._codec import encode_move, legal_indices
from lczerolens.evaluation import Evaluation
from lczerolens.evaluator import Evaluator
from lczerolens.schema import LczeroKeys
from lczerolens.search.capabilities import SearchAdapterCapability, require_adapter_capability
from lczerolens.search.limits import SearchLimit, Simulations
from lczerolens.search.result import SearchResult
from lczerolens.search.trace import (
    BackupUpdate,
    ChessPlayer,
    EdgeStatistics,
    EvaluatorCall,
    LeafRecord,
    NodeExpansion,
    PathStep,
    PositionEvaluation,
    RootAction,
    RootSelection,
    RootSnapshot,
    SearchBudget,
    SearchBudgetUnit,
    SearchCapability,
    SearchParameter,
    SearchProvenance,
    SearchTrace,
    SimulationEvent,
    ValuePerspective,
)


class _TensorEvaluator(Protocol):
    """The stable #130 evaluator shape consumed by reference search."""

    def __call__(self, board: chess.Board) -> TensorDict: ...


@dataclass
class _Edge:
    move: chess.Move
    prior: float
    visits: int = 0
    total_value: float = 0.0
    child: _Node | None = None

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


@dataclass
class _Node:
    board: chess.Board
    node_id: str
    edges: dict[str, _Edge] = field(default_factory=dict)
    expanded: bool = False


@dataclass(frozen=True)
class SemanticReplayResult:
    """Root result independently reconstructed from reference-search events."""

    root_statistics: tuple[EdgeStatistics, ...]
    root_policy: tuple[tuple[str, float], ...]
    selected_move: str


@dataclass(frozen=True)
class ReplayTolerance:
    """Numerical comparison policy recorded with a semantic replay audit."""

    relative: float = 1e-6
    absolute: float = 1e-6

    def __post_init__(self) -> None:
        if any(not math.isfinite(value) or value < 0 for value in (self.relative, self.absolute)):
            raise ValueError("Replay tolerances must be finite and non-negative.")


@dataclass(frozen=True)
class ReplayDiscrepancy:
    """One raw recorded-versus-replayed semantic difference."""

    event_id: str | None
    phase: str
    field: str
    recorded: object
    replayed: object


@dataclass(frozen=True)
class SemanticReplayCheckpoint:
    """Reconstructed tree and root state immediately after one event."""

    event_id: str
    simulation: int
    root_statistics: tuple[EdgeStatistics, ...]
    node_count: int
    expansion_count: int
    evaluator_call_count: int
    completed: bool
    discrepancies: tuple[ReplayDiscrepancy, ...]


@dataclass(frozen=True)
class SemanticReplayAudit:
    """Checkpoint-level replay evidence, including the first divergence."""

    tolerance: ReplayTolerance
    checkpoints: tuple[SemanticReplayCheckpoint, ...]
    first_divergence: ReplayDiscrepancy | None
    result: SemanticReplayResult | None


class SemanticReplayError(ValueError):
    """The first semantic divergence found while replaying a search trace."""

    def __init__(self, message: str, *, event_id: str | None = None, phase: str):
        self.event_id = event_id
        self.phase = phase
        location = f"Event {event_id} {phase}" if event_id is not None else phase
        super().__init__(f"{location}: {message}")


@dataclass(frozen=True)
class LeafEvaluationReplacement:
    """One explicit policy/value replacement at a recorded leaf event."""

    event_id: str
    value: float
    legal_policy_logits: tuple[tuple[str, float], ...]
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("A leaf-evaluation replacement requires a non-empty event ID.")
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise ValueError("Replacement value must be a real number.")
        value = float(self.value)
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("Replacement value must be finite and in [-1, 1].")
        object.__setattr__(self, "value", value)
        moves = tuple(move for move, _ in self.legal_policy_logits)
        if not moves or len(moves) != len(set(moves)) or moves != tuple(sorted(moves)):
            raise ValueError("Replacement legal-policy logits must have unique moves in sorted order.")
        if any(
            not isinstance(move, str)
            or isinstance(logit, bool)
            or not isinstance(logit, Real)
            or not math.isfinite(logit)
            for move, logit in self.legal_policy_logits
        ):
            raise ValueError("Replacement legal-policy logits must contain finite values for string moves.")
        object.__setattr__(
            self,
            "legal_policy_logits",
            tuple((move, float(logit)) for move, logit in self.legal_policy_logits),
        )
        dtype = getattr(torch, self.dtype, None)
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise ValueError(f"Unsupported replacement dtype {self.dtype!r}.")
        try:
            torch.device(self.device)
        except (RuntimeError, TypeError) as error:
            raise ValueError(f"Unsupported replacement device {self.device!r}.") from error

    @classmethod
    def from_event(cls, event: SimulationEvent) -> LeafEvaluationReplacement:
        """Build an exact no-op replacement from recorded evaluator evidence."""
        evaluator = event.leaf.evaluator
        value = event.leaf.evaluation.value
        if event.leaf.terminal or evaluator is None or value is None or evaluator.legal_policy_logits is None:
            raise ValueError(f"Event {event.event_id} has no replaceable leaf evaluation.")
        return cls(
            event.event_id,
            value,
            tuple(evaluator.legal_policy_logits),
            evaluator.dtype,
            evaluator.source_device,
        )


class CounterfactualReplayFormatError(ValueError):
    """Raised when leaf-replacement bytes are malformed or noncanonical."""


def serialize_leaf_evaluation_replacement(replacement: LeafEvaluationReplacement) -> bytes:
    """Return canonical versioned JSON bytes for one leaf replacement."""
    if not isinstance(replacement, LeafEvaluationReplacement):
        raise TypeError("serialize_leaf_evaluation_replacement expects a LeafEvaluationReplacement.")
    envelope = {
        "format": "lczerolens.leaf-evaluation-replacement",
        "format_version": 1,
        "replacement": {
            "device": replacement.device,
            "dtype": replacement.dtype,
            "event_id": replacement.event_id,
            "legal_policy_logits": [list(item) for item in replacement.legal_policy_logits],
            "value": replacement.value,
        },
    }
    try:
        return json.dumps(
            envelope,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (UnicodeEncodeError, ValueError) as error:
        raise CounterfactualReplayFormatError(f"Replacement is not canonical JSON: {error}") from error


def deserialize_leaf_evaluation_replacement(data: bytes) -> LeafEvaluationReplacement:
    """Restore one replacement while rejecting malformed or noncanonical bytes."""
    if not isinstance(data, bytes):
        raise TypeError("deserialize_leaf_evaluation_replacement expects bytes.")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, value in pairs:
            if key in record:
                raise CounterfactualReplayFormatError(f"Duplicate JSON field {key!r}.")
            record[key] = value
        return record

    try:
        envelope = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except CounterfactualReplayFormatError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CounterfactualReplayFormatError(f"Invalid replacement JSON: {error}") from error
    if not isinstance(envelope, dict) or set(envelope) != {"format", "format_version", "replacement"}:
        raise CounterfactualReplayFormatError("Replacement envelope has invalid fields.")
    version = envelope["format_version"]
    if envelope["format"] != "lczerolens.leaf-evaluation-replacement" or isinstance(version, bool) or version != 1:
        raise CounterfactualReplayFormatError("Unsupported replacement format or version.")
    record = envelope["replacement"]
    expected = {"device", "dtype", "event_id", "legal_policy_logits", "value"}
    if not isinstance(record, dict) or set(record) != expected:
        raise CounterfactualReplayFormatError("Replacement record has invalid fields.")
    logits = record["legal_policy_logits"]
    if not isinstance(logits, list) or any(not isinstance(item, list) or len(item) != 2 for item in logits):
        raise CounterfactualReplayFormatError("Replacement legal-policy logits have an invalid shape.")
    try:
        replacement = LeafEvaluationReplacement(
            event_id=record["event_id"],
            value=record["value"],
            legal_policy_logits=tuple((item[0], item[1]) for item in logits),
            dtype=record["dtype"],
            device=record["device"],
        )
    except (TypeError, ValueError) as error:
        raise CounterfactualReplayFormatError(f"Invalid replacement record: {error}") from error
    if serialize_leaf_evaluation_replacement(replacement) != data:
        raise CounterfactualReplayFormatError("Replacement bytes are valid JSON but are not canonical.")
    return replacement


def leaf_evaluation_replacement_digest(replacement: LeafEvaluationReplacement) -> str:
    """Return the SHA-256 digest of one canonical replacement."""
    return hashlib.sha256(serialize_leaf_evaluation_replacement(replacement)).hexdigest()


@dataclass(frozen=True)
class CounterfactualReplayResult:
    """A replayable counterfactual trace and its declared source relationship."""

    source_trace: SearchTrace
    replacement: LeafEvaluationReplacement
    restored_prefix_event_ids: tuple[str, ...]
    first_divergence_event_id: str | None
    trace: SearchTrace


@dataclass(frozen=True)
class RetainedEventReplayPlan:
    """An ordered, explicit selection of events from one replayable trace.

    ``events`` always follows the trace's original order, irrespective of the
    order in which callers supplied event IDs.  Events outside this plan make
    no contribution to the replayed result.
    """

    retained_event_ids: tuple[str, ...]
    omitted_event_ids: tuple[str, ...]
    events: tuple[SimulationEvent, ...]


@dataclass(frozen=True)
class RetainedEventReplayCosts:
    """Observed work represented by a retained-event replay plan."""

    simulations: int
    evaluator_calls: int
    expansions: int
    backup_updates: int
    path_steps: int


@dataclass(frozen=True)
class RetainedEventPath:
    """One selected event path without inferred structural additions."""

    event_id: str
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetainedEventFootprint:
    """Canonical topology and structural dependencies induced by selected events."""

    node_ids: tuple[str, ...]
    edges: tuple[PathStep, ...]
    paths: tuple[RetainedEventPath, ...]
    expansion_node_ids: tuple[str, ...]
    evaluator_call_node_ids: tuple[str, ...]
    ancestor_closed: bool
    missing_ancestor_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetainedEventReplayResult:
    """Root decision evidence reconstructed from a retained-event plan."""

    root_statistics: tuple[EdgeStatistics, ...]
    root_policy: tuple[tuple[str, float], ...]
    selected_move: str
    plan: RetainedEventReplayPlan
    costs: RetainedEventReplayCosts
    footprint: RetainedEventFootprint


class _ReferenceMCTS:
    """Sequential PUCT search whose output is a replayable :class:`SearchTrace`.

    The evaluator returns a single-board TensorDict with a raw 1858-logit
    ``policy`` and scalar ``value`` for the side to move.  Legal logits are
    masked and softmaxed over the stateless Lczero legal-policy mapping.
    """

    def __init__(self, c_puct: float = 1.0):
        if not math.isfinite(c_puct) or c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative.")
        self.c_puct = c_puct

    def search(
        self,
        board: chess.Board,
        evaluator: _TensorEvaluator | Callable[[chess.Board], TensorDict],
        simulations: int,
        *,
        root_evaluation: LeafEvaluationReplacement | None = None,
    ) -> SearchTrace:
        """Run a fixed number of simulations and return their full trace."""
        if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 1:
            raise ValueError("simulations must be a positive integer.")
        if board.is_game_over():
            raise ValueError("Reference search requires a non-terminal root position.")

        root = _Node(board.copy(stack=True), "node-0")
        root_evaluator_source = evaluator
        if root_evaluation is not None:
            if not isinstance(root_evaluation, LeafEvaluationReplacement):
                raise TypeError("Root evaluation must be a LeafEvaluationReplacement.")

            def replace_root_evaluation(board: chess.Board) -> TensorDict:
                return _replacement_output(board, root_evaluation)

            root_evaluator_source = replace_root_evaluation
        root_value, root_expansion, root_evaluator = self._expand(
            root, root_evaluator_source, ValuePerspective.ROOT_PLAYER
        )
        events, _ = self._run_simulations(root, evaluator, 0, simulations, 1)
        return self._trace(
            root,
            simulations,
            root_value,
            root_expansion,
            root_evaluator,
            events,
            provenance=self._provenance(root_evaluation),
        )

    def _run_simulations(
        self,
        root: _Node,
        evaluator: _TensorEvaluator | Callable[[chess.Board], TensorDict],
        start: int,
        stop: int,
        node_count: int,
        replacement: LeafEvaluationReplacement | None = None,
    ) -> tuple[list[SimulationEvent], int]:
        events: list[SimulationEvent] = []
        for simulation in range(start, stop):
            root_before = self._edge_stats(root, ValuePerspective.ROOT_PLAYER)
            node = root
            path: list[tuple[_Node, _Edge, _Node]] = []
            expansion: NodeExpansion | None = None

            while node.expanded and node.edges:
                edge = self._select(node)
                if edge.child is None:
                    child_board = node.board.copy(stack=True)
                    child_board.push(edge.move)
                    edge.child = _Node(child_board, f"node-{node_count}")
                    node_count += 1
                child = edge.child
                path.append((node, edge, child))
                node = child
                if not node.expanded:
                    break

            leaf_evaluator = evaluator
            if replacement is not None and simulation == start:

                def replace_leaf_evaluation(board: chess.Board) -> TensorDict:
                    return _replacement_output(board, replacement)

                leaf_evaluator = replace_leaf_evaluation
            leaf, expansion = self._leaf(node, leaf_evaluator, root.board.turn)
            backups = self._backup(path, leaf.evaluation.value or 0.0)
            root_after = self._edge_stats(root, ValuePerspective.ROOT_PLAYER)
            events.append(
                SimulationEvent(
                    event_id=f"simulation-{simulation}",
                    simulation=simulation,
                    path=tuple(
                        PathStep(parent.node_id, edge.move.uci(), child.node_id) for parent, edge, child in path
                    ),
                    leaf=leaf,
                    backups=tuple(backups),
                    expansion=expansion,
                    root_before=root_before,
                    root_after=root_after,
                )
            )
        return events, node_count

    def _trace(
        self,
        root: _Node,
        simulations: int,
        root_evaluation: PositionEvaluation,
        root_expansion: NodeExpansion,
        root_evaluator: EvaluatorCall,
        events: list[SimulationEvent],
        *,
        provenance: SearchProvenance,
    ) -> SearchTrace:
        actions = tuple(RootAction(edge) for edge in self._edge_stats(root, ValuePerspective.ROOT_PLAYER))
        selection = self._root_selection(actions)
        return SearchTrace(
            root_fen=root.board.fen(en_passant="fen"),
            root_player=ChessPlayer.WHITE if root.board.turn else ChessPlayer.BLACK,
            capability=SearchCapability.REPLAYABLE,
            provenance=provenance,
            snapshots=(
                RootSnapshot(
                    sequence=0,
                    budget=SearchBudget(SearchBudgetUnit.SIMULATIONS, requested=simulations, observed=simulations),
                    evaluation=root_evaluation,
                    selection=selection,
                    actions=actions,
                ),
            ),
            events=tuple(events),
            root_expansion=root_expansion,
            root_evaluator=root_evaluator,
            root_start_fen=root.board.root().fen(en_passant="fen"),
            root_move_history=tuple(move.uci() for move in root.board.move_stack),
        )

    def _provenance(self, root_replacement: LeafEvaluationReplacement | None) -> SearchProvenance:
        parameters = (
            SearchParameter("c_puct", self.c_puct),
            SearchParameter("selection", "Q + c_puct * P * sqrt(sum(N)) / (1 + N)"),
            SearchParameter("tie_break", "UCI lexicographic order"),
        )
        if root_replacement is not None:
            parameters += (
                SearchParameter("root_evaluation.event_id", root_replacement.event_id),
                SearchParameter(
                    "root_evaluation.replacement_sha256",
                    leaf_evaluation_replacement_digest(root_replacement),
                ),
            )
        return SearchProvenance(
            source="lczerolens-reference-mcts",
            engine="deterministic-reference",
            parameters=parameters,
        )

    def replay_counterfactual(
        self,
        trace: SearchTrace,
        evaluator: _TensorEvaluator | Callable[[chess.Board], TensorDict],
        replacement: LeafEvaluationReplacement,
    ) -> CounterfactualReplayResult:
        """Restore a validated prefix and re-execute the original remaining budget."""
        if not isinstance(trace, SearchTrace):
            raise TypeError("Counterfactual replay requires a SearchTrace.")
        if not isinstance(replacement, LeafEvaluationReplacement):
            raise TypeError("Counterfactual replay requires a LeafEvaluationReplacement.")
        replay_search_trace(trace)
        events = trace.events or ()
        by_id = {event.event_id: index for index, event in enumerate(events)}
        if replacement.event_id not in by_id:
            raise ValueError(f"Unknown replacement event ID {replacement.event_id!r}.")
        target_index = by_id[replacement.event_id]
        target = events[target_index]
        if target.leaf.terminal or target.leaf.evaluator is None:
            raise ValueError(f"Event {replacement.event_id} has no replaceable leaf evaluation.")
        if not math.isclose(self.c_puct, _replay_c_puct(trace), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("Counterfactual replay c_puct must match the source trace.")

        # An exact recorded replacement is the identity intervention. Restore
        # the target event semantically as part of the certified prefix rather
        # than round-tripping its logits through a backend softmax kernel: two
        # mathematically identical softmax calls can differ by an ulp across
        # tensor layouts. The remaining suffix is still selected and evaluated
        # live, so this is a genuine prefix-resume check rather than returning
        # the source trace unchanged.
        identity_replacement = replacement == LeafEvaluationReplacement.from_event(target)

        root_board = _replay_root_board(trace)
        root_id = events[0].path[0].node_id
        root = _Node(root_board, root_id)
        _initialize_replay_root(root, trace, events[0])
        nodes = {root_id: root}
        for event in events[:target_index]:
            path = _replay_path(root, nodes, event, self.c_puct)
            _replay_leaf(path[-1][2], event, root.board.turn)
            _replay_backups(path, event)
        resume_index = target_index
        if identity_replacement:
            path = _replay_path(root, nodes, target, self.c_puct)
            _replay_leaf(path[-1][2], target, root.board.turn)
            _replay_backups(path, target)
            resume_index += 1
        node_count = _canonical_replay_node_count(nodes)

        suffix, _ = self._run_simulations(
            root,
            evaluator,
            resume_index,
            len(events),
            node_count,
            None if identity_replacement else replacement,
        )
        counterfactual = self._trace(
            root,
            len(events),
            trace.snapshots[0].evaluation,
            trace.root_expansion,
            trace.root_evaluator,
            [*events[:resume_index], *suffix],
            provenance=trace.provenance,
        )
        replay_search_trace(counterfactual)
        first_divergence = next(
            (
                original.event_id
                for original, changed in zip(events, counterfactual.events or ())
                if original != changed
            ),
            None,
        )
        return CounterfactualReplayResult(
            trace,
            replacement,
            tuple(event.event_id for event in events[:resume_index]),
            first_divergence,
            counterfactual,
        )

    def _expand(
        self,
        node: _Node,
        evaluator: _TensorEvaluator | Callable[[chess.Board], TensorDict],
        perspective: ValuePerspective,
    ) -> tuple[PositionEvaluation, NodeExpansion, EvaluatorCall]:
        output = self._single_evaluation(evaluator(node.board))
        policy = output.get("policy")
        value = output.get("value")
        if not isinstance(policy, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError("Evaluator must return TensorDict tensors named 'policy' and 'value'.")
        if value.numel() != 1 or not torch.isfinite(value).all() or not -1 <= value.item() <= 1:
            raise ValueError("Evaluator value must be one finite scalar in [-1, 1].")
        if policy.ndim != 1 or policy.shape[0] != 1858:
            raise ValueError("Evaluator policy must contain exactly 1858 raw logits.")
        legal_moves = sorted(node.board.legal_moves, key=lambda move: move.uci())
        indices = legal_indices(node.board).to(policy.device)
        legal_logits = policy.detach().gather(0, indices)
        if not torch.isfinite(legal_logits).all():
            raise ValueError("Evaluator policy must not contain non-finite legal logits.")
        priors = torch.softmax(legal_logits, dim=0)
        index_values = indices.detach().cpu().tolist()
        by_index = {index: prior.item() for index, prior in zip(index_values, priors)}
        node.edges = {move.uci(): _Edge(move, by_index[encode_move(node.board, move)]) for move in legal_moves}
        node.expanded = True
        edge_stats = self._edge_stats(node, perspective)
        by_index = {index: logit.item() for index, logit in zip(index_values, legal_logits)}
        evaluator_call = EvaluatorCall(
            dtype=str(policy.dtype).removeprefix("torch."),
            source_device=str(policy.device),
            search_device=str(policy.device),
            legal_policy_logits=tuple(
                (move.uci(), float(by_index[encode_move(node.board, move)])) for move in legal_moves
            ),
        )
        return (
            PositionEvaluation(perspective, value=float(value.item())),
            NodeExpansion(node.node_id, edge_stats),
            evaluator_call,
        )

    @staticmethod
    def _single_evaluation(output: TensorDict) -> TensorDict:
        """Normalize the canonical evaluator's optional singleton batch."""
        if not isinstance(output, TensorDict):
            raise ValueError("Evaluator must return a TensorDict.")
        if output.batch_size == torch.Size([]):
            return output
        if output.batch_size == torch.Size([1]):
            return output[0]
        raise ValueError("Reference search requires one evaluator result per board.")

    def _leaf(
        self,
        node: _Node,
        evaluator: _TensorEvaluator | Callable[[chess.Board], TensorDict],
        root_turn: chess.Color,
    ) -> tuple[LeafRecord, NodeExpansion | None]:
        perspective = ValuePerspective.ROOT_PLAYER if node.board.turn == root_turn else ValuePerspective.SIDE_TO_MOVE
        if node.board.is_game_over():
            outcome = node.board.outcome()
            value = (
                0.0
                if outcome is None or outcome.winner is None
                else 1.0
                if outcome.winner == node.board.turn
                else -1.0
            )
            return LeafRecord(node.node_id, PositionEvaluation(perspective, value=value), True), None
        evaluation, expansion, evaluator_call = self._expand(node, evaluator, perspective)
        return (
            LeafRecord(node.node_id, evaluation, False, evaluator_call),
            expansion,
        )

    def _select(self, node: _Node) -> _Edge:
        total_visits = sum(edge.visits for edge in node.edges.values())
        scale = math.sqrt(total_visits)
        scores = {
            move: edge.mean_value + self.c_puct * edge.prior * scale / (1 + edge.visits)
            for move, edge in node.edges.items()
        }
        best_score = max(scores.values())
        return node.edges[min(move for move, score in scores.items() if score == best_score)]

    def _backup(self, path: list[tuple[_Node, _Edge, _Node]], leaf_value: float) -> list[BackupUpdate]:
        value = leaf_value
        updates: list[BackupUpdate] = []
        for parent, edge, _ in reversed(path):
            value = -value
            perspective = ValuePerspective.ROOT_PLAYER if parent.node_id == "node-0" else ValuePerspective.SIDE_TO_MOVE
            before = self._edge_stat(edge, perspective)
            edge.visits += 1
            edge.total_value += value
            after = self._edge_stat(edge, perspective)
            updates.append(BackupUpdate(parent.node_id, value, before, after))
        return updates

    @staticmethod
    def _edge_stat(edge: _Edge, perspective: ValuePerspective) -> EdgeStatistics:
        return EdgeStatistics(edge.move.uci(), perspective, edge.prior, edge.visits, edge.total_value, edge.mean_value)

    def _edge_stats(self, node: _Node, perspective: ValuePerspective) -> tuple[EdgeStatistics, ...]:
        return tuple(self._edge_stat(node.edges[move], perspective) for move in sorted(node.edges))

    @staticmethod
    def _root_selection(actions: tuple[RootAction, ...]) -> RootSelection:
        best_visits = max(action.statistics.visits or 0 for action in actions)
        move = min(action.statistics.move for action in actions if (action.statistics.visits or 0) == best_visits)
        return RootSelection(move, rule="maximum visit count", tie_break="UCI lexicographic order", temperature=0.0)


class ReferenceSearch:
    """Deterministic, auditable search behind the shared ``run`` interface.

    This producer intentionally models neither production Lczero behavior nor
    its batching, FPU, collisions, pruning, transpositions, or time management.
    """

    def __init__(self, evaluator: Evaluator, *, c_puct: float = 1.0):
        if not isinstance(evaluator, Evaluator):
            raise TypeError("ReferenceSearch requires an Evaluator.")
        self.evaluator = evaluator
        self._core = _ReferenceMCTS(c_puct)

    def supports(self, capability: SearchAdapterCapability) -> bool:
        """Return whether this adapter implements an optional search input."""
        if not isinstance(capability, SearchAdapterCapability):
            raise TypeError("Search adapter capability checks require a SearchAdapterCapability value.")
        return capability is SearchAdapterCapability.ROOT_EVALUATION_REPLACEMENT

    def require(self, capability: SearchAdapterCapability) -> ReferenceSearch:
        """Require an optional input capability and return this adapter."""
        require_adapter_capability(frozenset({SearchAdapterCapability.ROOT_EVALUATION_REPLACEMENT}), capability)
        return self

    def run(
        self,
        board: chess.Board,
        limit: SearchLimit,
        *,
        root_evaluation: LeafEvaluationReplacement | None = None,
    ) -> SearchResult:
        """Run exactly the requested simulations on a plain chess board."""
        if not isinstance(board, chess.Board):
            raise TypeError("ReferenceSearch.run requires a python-chess Board.")
        if not isinstance(limit, Simulations):
            raise ValueError("ReferenceSearch supports only Simulations limits.")
        _validate_reference_board(board)
        trace = self._core.search(
            board.copy(stack=True), self._evaluate, simulations=limit.value, root_evaluation=root_evaluation
        )
        return SearchResult.from_trace(trace)

    def replay_counterfactual(
        self, trace: SearchTrace, replacement: LeafEvaluationReplacement
    ) -> CounterfactualReplayResult:
        """Apply one leaf replacement and deterministically rerun the original suffix."""
        return self._core.replay_counterfactual(trace, self._evaluate, replacement)

    def _evaluate(self, board: chess.Board) -> TensorDict:
        evaluated = self.evaluator.evaluate(board)
        if not isinstance(evaluated, Evaluation):
            raise TypeError("ReferenceSearch requires one Evaluation per position.")
        value = evaluated.value
        if value is None:
            raise ValueError("ReferenceSearch requires a native or explicitly derived scalar value.")
        policy = evaluated.tensors[LczeroKeys.NETWORK_POLICY_LOGITS]
        return TensorDict(
            {
                "policy": policy,
                "value": torch.as_tensor(value.value, dtype=policy.dtype, device=policy.device),
            },
            batch_size=[],
        )


def _replacement_output(board: chess.Board, replacement: LeafEvaluationReplacement) -> TensorDict:
    legal_moves = tuple(sorted((move.uci() for move in board.legal_moves)))
    replacement_moves = tuple(move for move, _ in replacement.legal_policy_logits)
    if replacement_moves != legal_moves:
        raise ValueError(
            f"Replacement event {replacement.event_id} legal moves do not match the selected leaf position."
        )
    dtype = getattr(torch, replacement.dtype)
    device = torch.device(replacement.device)
    policy = torch.zeros(1858, dtype=dtype, device=device)
    for move_uci, logit in replacement.legal_policy_logits:
        move = chess.Move.from_uci(move_uci)
        policy[encode_move(board, move)] = logit
    return TensorDict(
        {
            "policy": policy,
            "value": torch.tensor(replacement.value, dtype=dtype, device=device),
        },
        batch_size=[],
    )


def _canonical_replay_node_count(nodes: dict[str, _Node]) -> int:
    """Return the next node index after validating a restored reference tree."""
    expected = {f"node-{index}" for index in range(len(nodes))}
    if nodes.keys() != expected:
        raise SemanticReplayError("restored prefix has non-canonical node IDs.", phase="prefix")
    return len(nodes)


def _validate_reference_board(board: chess.Board) -> None:
    if board.uci_variant != "chess" or board.chess960:
        raise ValueError("ReferenceSearch currently supports standard chess positions.")


def replay_root_events(events: tuple[SimulationEvent, ...]) -> tuple[EdgeStatistics, ...]:
    """Reconstruct final root state from events without accessing a search tree."""
    if not events:
        raise ValueError("At least one simulation event is required.")
    initial = events[0].root_before or ()
    state = {edge.move: edge for edge in initial}
    if len(state) != len(initial):
        raise ValueError("Replayable events require unique root moves.")
    if not state:
        raise ValueError("Replayable events require a non-empty root state.")
    for event in events:
        before = {edge.move: edge for edge in event.root_before or ()}
        after = {edge.move: edge for edge in event.root_after or ()}
        if len(before) != len(event.root_before or ()) or len(after) != len(event.root_after or ()):
            raise ValueError(f"Event {event.event_id} has duplicate root moves.")
        if before.keys() != after.keys():
            raise ValueError(f"Event {event.event_id} changes the root move set.")
        if before != state:
            raise ValueError(f"Event {event.event_id} does not chain from the reconstructed root state.")
        changed = [move for move in state if state[move] != after.get(move)]
        if len(changed) != 1:
            raise ValueError(f"Event {event.event_id} must change exactly one root edge.")
        move = changed[0]
        if not any(update.before == state[move] and update.after == after[move] for update in event.backups):
            raise ValueError(f"Event {event.event_id} has no matching root backup.")
        state = after
    return tuple(state[move] for move in sorted(state))


def plan_retained_events(trace: SearchTrace, event_ids: tuple[str, ...] | None = None) -> RetainedEventReplayPlan:
    """Resolve an explicit retained-event selection in original trace order.

    An empty tuple is an intentional empty selection; ``None`` selects every
    event.  The plan does not add structural events: each selected event must
    carry the root transition needed to account for its own backup.
    """
    trace.require(SearchCapability.REPLAYABLE)
    events = trace.events or ()
    known = {event.event_id: event for event in events}
    if len(known) != len(events):
        raise ValueError("Replayable traces require unique event IDs.")
    requested = tuple(known) if event_ids is None else tuple(event_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("Retained event IDs must be unique.")
    unknown = sorted(set(requested) - known.keys())
    if unknown:
        raise ValueError(f"Unknown retained event IDs: {', '.join(unknown)}.")
    retained = frozenset(requested)
    selected = tuple(event for event in events if event.event_id in retained)
    return RetainedEventReplayPlan(
        retained_event_ids=tuple(event.event_id for event in selected),
        omitted_event_ids=tuple(event.event_id for event in events if event.event_id not in retained),
        events=selected,
    )


def replay_retained_events(trace: SearchTrace, event_ids: tuple[str, ...] | None = None) -> RetainedEventReplayResult:
    """Replay only retained root-backup contributions from a trace.

    Each retained event contributes its recorded root backup delta to the
    initialized root state.  Omitted events are neither replayed as structural
    prerequisites nor used to restore visits or values, which lets sparse and
    non-contiguous selections remain explicit counterfactuals.
    """
    plan = plan_retained_events(trace, event_ids)
    events = trace.events or ()
    if not events:
        raise ValueError("Replayable traces require at least one simulation event.")
    initial = _retained_initial_root_state(events[0])
    state = {edge.move: edge for edge in initial}
    for event in plan.events:
        move, before, after = _retained_root_transition(event, set(state))
        state[move] = _apply_retained_root_delta(state[move], before, after, event)
    root_statistics = tuple(replace(state[move], exploration=None) for move in sorted(state))
    total_visits = sum(edge.visits or 0 for edge in root_statistics)
    root_policy = tuple(
        (edge.move, 0.0 if total_visits == 0 else (edge.visits or 0) / total_visits) for edge in root_statistics
    )
    best_visits = max(edge.visits or 0 for edge in root_statistics)
    selected_move = min(edge.move for edge in root_statistics if (edge.visits or 0) == best_visits)
    costs = RetainedEventReplayCosts(
        simulations=len(plan.events),
        evaluator_calls=sum(event.leaf.evaluator is not None for event in plan.events),
        expansions=sum(event.expansion is not None for event in plan.events),
        backup_updates=sum(len(event.backups) for event in plan.events),
        path_steps=sum(len(event.path) for event in plan.events),
    )
    footprint = _retained_event_footprint(events, plan)
    result = RetainedEventReplayResult(root_statistics, root_policy, selected_move, plan, costs, footprint)
    if len(plan.events) == len(events):
        final = trace.snapshots[-1]
        final_statistics = tuple(action.statistics for action in final.actions or ())
        if (
            len(result.root_statistics) != len(final_statistics)
            or any(
                not _same_retained_edge(left, right) for left, right in zip(result.root_statistics, final_statistics)
            )
            or final.selection is None
            or result.selected_move != final.selection.move
        ):
            raise ValueError("Full retained-event replay diverges from the recorded root result.")
    return result


def _retained_event_footprint(
    events: tuple[SimulationEvent, ...], plan: RetainedEventReplayPlan
) -> RetainedEventFootprint:
    """Describe exactly the topology present in ``plan`` and its omitted prerequisites."""
    retained = set(plan.retained_event_ids)
    expansion_owner = {event.expansion.node_id: event.event_id for event in events if event.expansion is not None}
    root_id = events[0].path[0].node_id if events and events[0].path else None
    nodes: set[str] = set()
    edges: dict[tuple[str, str, str], PathStep] = {}
    paths: list[RetainedEventPath] = []
    expanded: set[str] = set()
    evaluated: set[str] = set()
    missing: set[str] = set()
    for event in plan.events:
        path_nodes: list[str] = []
        for index, step in enumerate(event.path):
            if index == 0:
                path_nodes.append(step.node_id)
            path_nodes.append(step.child_id)
            nodes.update((step.node_id, step.child_id))
            edges[(step.node_id, step.move, step.child_id)] = step
            owner = expansion_owner.get(step.node_id)
            if step.node_id != root_id and owner is not None and owner not in retained:
                missing.add(owner)
        nodes.add(event.leaf.node_id)
        paths.append(RetainedEventPath(event.event_id, tuple(path_nodes)))
        if event.expansion is not None:
            nodes.add(event.expansion.node_id)
            expanded.add(event.expansion.node_id)
        if event.leaf.evaluator is not None:
            evaluated.add(event.leaf.node_id)
    return RetainedEventFootprint(
        node_ids=tuple(sorted(nodes)),
        edges=tuple(edges[key] for key in sorted(edges)),
        paths=tuple(paths),
        expansion_node_ids=tuple(sorted(expanded)),
        evaluator_call_node_ids=tuple(sorted(evaluated)),
        ancestor_closed=not missing,
        missing_ancestor_event_ids=tuple(event.event_id for event in events if event.event_id in missing),
    )


def _retained_initial_root_state(event: SimulationEvent) -> tuple[EdgeStatistics, ...]:
    initial = event.root_before
    if not initial:
        raise ValueError(f"Event {event.event_id} needs a non-empty initial root state.")
    state = {edge.move: edge for edge in initial}
    if len(state) != len(initial):
        raise ValueError(f"Event {event.event_id} has duplicate root moves.")
    return initial


def _retained_root_transition(
    event: SimulationEvent, expected_moves: set[str]
) -> tuple[str, EdgeStatistics, EdgeStatistics]:
    before = {edge.move: edge for edge in event.root_before or ()}
    after = {edge.move: edge for edge in event.root_after or ()}
    if len(before) != len(event.root_before or ()) or len(after) != len(event.root_after or ()):
        raise ValueError(f"Event {event.event_id} has duplicate root moves.")
    if before.keys() != after.keys() or before.keys() != expected_moves:
        raise ValueError(f"Event {event.event_id} changes the root move set.")
    changed = [move for move in before if before[move] != after[move]]
    if len(changed) != 1:
        raise ValueError(f"Event {event.event_id} must change exactly one root edge.")
    move = changed[0]
    if not any(update.before == before[move] and update.after == after[move] for update in event.backups):
        raise ValueError(f"Event {event.event_id} has no matching root backup.")
    return move, before[move], after[move]


def _apply_retained_root_delta(
    current: EdgeStatistics, before: EdgeStatistics, after: EdgeStatistics, event: SimulationEvent
) -> EdgeStatistics:
    if current.move != before.move or current.perspective is not before.perspective:
        raise ValueError(f"Event {event.event_id} has incompatible root-edge evidence.")
    if current.prior != before.prior or after.prior != before.prior:
        raise ValueError(f"Event {event.event_id} changes a root prior.")
    before_visits, after_visits = before.visits, after.visits
    before_value, after_value = before.total_value, after.total_value
    if before_visits is None or after_visits is None or before_value is None or after_value is None:
        raise ValueError(f"Event {event.event_id} needs root visit and value evidence.")
    visits = (current.visits or 0) + after_visits - before_visits
    total_value = (current.total_value or 0.0) + after_value - before_value
    if visits < 0:
        raise ValueError(f"Event {event.event_id} would make root visits negative.")
    return EdgeStatistics(
        move=current.move,
        perspective=current.perspective,
        prior=current.prior,
        visits=visits,
        total_value=total_value,
        mean_value=0.0 if visits == 0 else total_value / visits,
    )


def audit_search_trace(trace: SearchTrace, tolerance: ReplayTolerance | None = None) -> SemanticReplayAudit:
    """Audit deterministic-reference replay at every event checkpoint.

    Unlike :func:`replay_root_events`, this reconstructs tree state from the
    root position and semantic event records. Recorded root states are compared
    field by field but are never used to advance the reconstructed tree.
    """
    tolerance = tolerance or ReplayTolerance()
    if not isinstance(tolerance, ReplayTolerance):
        raise TypeError("tolerance must be ReplayTolerance.")
    trace.require(SearchCapability.REPLAYABLE)
    if trace.provenance.source != "lczerolens-reference-mcts" or trace.provenance.engine != "deterministic-reference":
        raise SemanticReplayError(
            "semantic replay is limited to deterministic ReferenceSearch traces.",
            phase="provenance",
        )
    events = trace.events or ()
    if not events:
        raise SemanticReplayError("at least one simulation event is required.", phase="events")
    c_puct = _replay_c_puct(trace)
    root_board = _replay_root_board(trace)
    first = events[0]
    if not first.path:
        raise SemanticReplayError(
            "a non-terminal root simulation needs a path.", event_id=first.event_id, phase="path"
        )
    root_id = first.path[0].node_id
    root = _Node(root_board, root_id)
    _initialize_replay_root(root, trace, first, tolerance)
    nodes = {root_id: root}
    checkpoints: list[SemanticReplayCheckpoint] = []
    first_divergence: ReplayDiscrepancy | None = None
    evaluator_calls = 1

    for expected_simulation, event in enumerate(events):
        if event.simulation != expected_simulation:
            raise SemanticReplayError(
                f"expected simulation {expected_simulation}, got {event.simulation}.",
                event_id=event.event_id,
                phase="sequence",
            )
        before = _replay_edge_stats(root, ValuePerspective.ROOT_PLAYER)
        discrepancies = list(_edge_sequence_discrepancies(event.root_before, before, event, "root_before", tolerance))
        try:
            path = _replay_path(root, nodes, event, c_puct)
            leaf_node = path[-1][2]
            _replay_leaf(leaf_node, event, root.board.turn, tolerance)
            if event.leaf.evaluator is not None:
                evaluator_calls += 1
            _replay_backups(path, event, tolerance)
        except SemanticReplayError as error:
            discrepancies.append(ReplayDiscrepancy(event.event_id, error.phase, "semantic_error", str(error), None))
            if first_divergence is None:
                first_divergence = discrepancies[0]
            checkpoints.append(
                SemanticReplayCheckpoint(
                    event.event_id,
                    event.simulation,
                    _replay_edge_stats(root, ValuePerspective.ROOT_PLAYER),
                    len(nodes),
                    sum(node.expanded for node in nodes.values()),
                    evaluator_calls,
                    False,
                    tuple(discrepancies),
                )
            )
            break
        after = _replay_edge_stats(root, ValuePerspective.ROOT_PLAYER)
        discrepancies.extend(_edge_sequence_discrepancies(event.root_after, after, event, "root_after", tolerance))
        if discrepancies and first_divergence is None:
            first_divergence = discrepancies[0]
        checkpoints.append(
            SemanticReplayCheckpoint(
                event.event_id,
                event.simulation,
                after,
                len(nodes),
                sum(node.expanded for node in nodes.values()),
                evaluator_calls,
                True,
                tuple(discrepancies),
            )
        )

    root_statistics = _replay_edge_stats(root, ValuePerspective.ROOT_PLAYER)
    total_visits = sum(edge.visits or 0 for edge in root_statistics)
    root_policy = tuple(
        (edge.move, 0.0 if total_visits == 0 else (edge.visits or 0) / total_visits) for edge in root_statistics
    )
    best_visits = max(edge.visits or 0 for edge in root_statistics)
    selected_move = min(edge.move for edge in root_statistics if (edge.visits or 0) == best_visits)
    result = SemanticReplayResult(root_statistics, root_policy, selected_move)
    if len(checkpoints) == len(events):
        final = trace.snapshots[-1]
        final_discrepancies = list(
            _edge_sequence_discrepancies(
                tuple(action.statistics for action in final.actions or ()),
                root_statistics,
                None,
                "result",
                tolerance,
            )
        )
        if final.selection is None or final.selection.move != selected_move:
            final_discrepancies.append(
                ReplayDiscrepancy(
                    None,
                    "result",
                    "selected_move",
                    None if final.selection is None else final.selection.move,
                    selected_move,
                )
            )
        if final_discrepancies:
            last = checkpoints[-1]
            checkpoints[-1] = replace(last, discrepancies=last.discrepancies + tuple(final_discrepancies))
            if first_divergence is None:
                first_divergence = final_discrepancies[0]
    return SemanticReplayAudit(
        tolerance,
        tuple(checkpoints),
        first_divergence,
        result if first_divergence is None and len(checkpoints) == len(events) else None,
    )


def replay_search_trace(trace: SearchTrace) -> SemanticReplayResult:
    """Independently replay a trace, raising at its first semantic divergence."""
    audit = audit_search_trace(trace)
    if audit.first_divergence is not None:
        divergence = audit.first_divergence
        message = f"{divergence.field} differs: recorded={divergence.recorded!r}, replayed={divergence.replayed!r}."
        if divergence.field == "semantic_error":
            message = str(divergence.recorded).split(": ", 1)[-1]
        elif divergence.phase in {"root_before", "root_after"}:
            message = "recorded root state diverges from replay."
        elif divergence.phase == "result" and divergence.field == "selected_move":
            message = "selected move diverges from replay."
        elif divergence.phase == "result":
            message = "final root action statistics diverge from replay."
        raise SemanticReplayError(
            message,
            event_id=divergence.event_id,
            phase=divergence.phase,
        )
    assert audit.result is not None
    return audit.result


def _replay_c_puct(trace: SearchTrace) -> float:
    values = [parameter.value for parameter in trace.provenance.parameters if parameter.name == "c_puct"]
    if len(values) != 1 or isinstance(values[0], bool) or not isinstance(values[0], (int, float)):
        raise SemanticReplayError("provenance needs one numeric c_puct parameter.", phase="provenance")
    value = float(values[0])
    if not math.isfinite(value) or value < 0:
        raise SemanticReplayError("c_puct must be finite and non-negative.", phase="provenance")
    return value


def _replay_root_board(trace: SearchTrace) -> chess.Board:
    if trace.root_start_fen is None or trace.root_move_history is None:
        raise SemanticReplayError(
            "reference traces need root history to replay history-dependent terminality.",
            phase="root_history",
        )
    board = chess.Board(trace.root_start_fen)
    for move_uci in trace.root_move_history:
        board.push_uci(move_uci)
    if board.fen(en_passant="fen") != trace.root_fen:
        raise SemanticReplayError("root history does not reconstruct root_fen.", phase="root_history")
    return board


def _initialize_replay_root(
    root: _Node, trace: SearchTrace, event: SimulationEvent, tolerance: ReplayTolerance | None = None
) -> None:
    initial = event.root_before or ()
    legal = {move.uci(): move for move in root.board.legal_moves}
    expansion = trace.root_expansion
    evaluator = trace.root_evaluator
    if expansion is None or evaluator is None or expansion.node_id != root.node_id:
        raise SemanticReplayError(
            "reference trace needs matching root expansion and evaluator evidence.", phase="root_expansion"
        )
    expanded = {edge.move: edge for edge in expansion.edges}
    logits = dict(evaluator.legal_policy_logits or ())
    if expanded.keys() != legal.keys() or logits.keys() != legal.keys():
        raise SemanticReplayError(
            "expanded root edges and evaluator logits must match the legal moves.", phase="root_expansion"
        )
    priors = _priors_from_logits(logits, evaluator.dtype)
    if {edge.move for edge in initial} != legal.keys():
        raise SemanticReplayError(
            "initial root edges do not match the legal moves.", event_id=event.event_id, phase="root_before"
        )
    for move in sorted(legal):
        edge = expanded[move]
        if (
            edge.perspective is not ValuePerspective.ROOT_PLAYER
            or edge.prior is None
            or (edge.visits, edge.total_value, edge.mean_value) != (0, 0.0, 0.0)
            or not _close(edge.prior, priors[move], tolerance)
        ):
            raise SemanticReplayError(
                f"root edge {move} does not match the evaluator-derived initial state.", phase="root_expansion"
            )
        # Validate against evaluator logits above, then retain the producer's
        # exact serialized float so prefix-resume does not introduce an ulp.
        root.edges[move] = _Edge(legal[move], edge.prior)
    for edge in initial:
        if edge.perspective is not ValuePerspective.ROOT_PLAYER:
            raise SemanticReplayError(
                f"root edge {edge.move} has perspective {edge.perspective.value}.",
                event_id=event.event_id,
                phase="root_before",
            )
        if (
            edge.prior is None
            or (edge.visits, edge.total_value, edge.mean_value) != (0, 0.0, 0.0)
            or not _close(edge.prior, root.edges[edge.move].prior, tolerance)
        ):
            raise SemanticReplayError(
                f"root edge {edge.move} is not an unvisited initialized edge.",
                event_id=event.event_id,
                phase="root_before",
            )
    root.expanded = True


def _replay_path(
    root: _Node, nodes: dict[str, _Node], event: SimulationEvent, c_puct: float
) -> list[tuple[_Node, _Edge, _Node]]:
    node = root
    replayed: list[tuple[_Node, _Edge, _Node]] = []
    for depth, recorded in enumerate(event.path):
        if not node.expanded or not node.edges:
            raise SemanticReplayError(
                f"path continues beyond unexpanded node {node.node_id} at depth {depth}.",
                event_id=event.event_id,
                phase="path",
            )
        expected_edge = _replay_select(node, c_puct)
        if recorded.node_id != node.node_id or recorded.move != expected_edge.move.uci():
            raise SemanticReplayError(
                f"depth {depth} expected {node.node_id}:{expected_edge.move.uci()}, "
                f"got {recorded.node_id}:{recorded.move}.",
                event_id=event.event_id,
                phase="path",
            )
        if expected_edge.child is None:
            if recorded.child_id in nodes:
                raise SemanticReplayError(
                    f"new child ID {recorded.child_id!r} is already in use.",
                    event_id=event.event_id,
                    phase="path",
                )
            child_board = node.board.copy(stack=True)
            child_board.push(expected_edge.move)
            expected_edge.child = _Node(child_board, recorded.child_id)
            nodes[recorded.child_id] = expected_edge.child
        elif expected_edge.child.node_id != recorded.child_id:
            raise SemanticReplayError(
                f"edge {recorded.move} points to {expected_edge.child.node_id!r}, not {recorded.child_id!r}.",
                event_id=event.event_id,
                phase="path",
            )
        child = expected_edge.child
        replayed.append((node, expected_edge, child))
        node = child
        if not node.expanded:
            if depth != len(event.path) - 1:
                raise SemanticReplayError(
                    f"path continues after first unexpanded node {node.node_id}.",
                    event_id=event.event_id,
                    phase="path",
                )
            break
    if not replayed:
        raise SemanticReplayError("simulation path is empty.", event_id=event.event_id, phase="path")
    if node.expanded:
        raise SemanticReplayError(
            f"path ends early at already expanded node {node.node_id}.",
            event_id=event.event_id,
            phase="path",
        )
    return replayed


def _replay_select(node: _Node, c_puct: float) -> _Edge:
    scale = math.sqrt(sum(edge.visits for edge in node.edges.values()))
    scores = {
        move: edge.mean_value + c_puct * edge.prior * scale / (1 + edge.visits) for move, edge in node.edges.items()
    }
    best = max(scores.values())
    return node.edges[min(move for move, score in scores.items() if score == best)]


def _replay_leaf(
    node: _Node,
    event: SimulationEvent,
    root_turn: chess.Color,
    tolerance: ReplayTolerance | None = None,
) -> None:
    if event.leaf.node_id != node.node_id:
        raise SemanticReplayError(
            f"path ends at {node.node_id!r}, but leaf is {event.leaf.node_id!r}.",
            event_id=event.event_id,
            phase="leaf",
        )
    terminal = node.board.is_game_over()
    if event.leaf.terminal is not terminal:
        raise SemanticReplayError(
            "terminal flag disagrees with the replayed board.", event_id=event.event_id, phase="leaf"
        )
    perspective = ValuePerspective.ROOT_PLAYER if node.board.turn == root_turn else ValuePerspective.SIDE_TO_MOVE
    if event.leaf.evaluation.perspective is not perspective:
        raise SemanticReplayError(
            f"expected {perspective.value} evaluation perspective.", event_id=event.event_id, phase="leaf"
        )
    if terminal:
        if event.expansion is not None or event.leaf.evaluator is not None:
            raise SemanticReplayError(
                "terminal leaves cannot have an expansion or evaluator call.", event_id=event.event_id, phase="leaf"
            )
        outcome = node.board.outcome()
        expected = (
            0.0 if outcome is None or outcome.winner is None else 1.0 if outcome.winner == node.board.turn else -1.0
        )
        if not _close(event.leaf.evaluation.value, expected, tolerance):
            raise SemanticReplayError(f"terminal value should be {expected}.", event_id=event.event_id, phase="leaf")
        return
    if event.expansion is None or event.leaf.evaluator is None:
        raise SemanticReplayError(
            "non-terminal first visits need expansion and evaluator evidence.",
            event_id=event.event_id,
            phase="expansion",
        )
    _replay_expansion(node, event, perspective, tolerance)


def _replay_expansion(
    node: _Node,
    event: SimulationEvent,
    perspective: ValuePerspective,
    tolerance: ReplayTolerance | None = None,
) -> None:
    expansion = event.expansion
    evaluator = event.leaf.evaluator
    assert expansion is not None and evaluator is not None
    if expansion.node_id != node.node_id:
        raise SemanticReplayError(
            "expansion node does not match the leaf.", event_id=event.event_id, phase="expansion"
        )
    legal = {move.uci(): move for move in node.board.legal_moves}
    edges = {edge.move: edge for edge in expansion.edges}
    logits = dict(evaluator.legal_policy_logits or ())
    if edges.keys() != legal.keys() or logits.keys() != legal.keys():
        raise SemanticReplayError(
            "expanded edges and evaluator logits must match the legal moves.",
            event_id=event.event_id,
            phase="expansion",
        )
    priors = _priors_from_logits(logits, evaluator.dtype)
    ordered = sorted(legal)
    for move in ordered:
        edge = edges[move]
        if edge.perspective is not perspective or edge.prior is None:
            raise SemanticReplayError(
                f"edge {move} has the wrong perspective or no prior.",
                event_id=event.event_id,
                phase="expansion",
            )
        if (edge.visits, edge.total_value, edge.mean_value) != (0, 0.0, 0.0) or not _close(
            edge.prior, priors[move], tolerance
        ):
            raise SemanticReplayError(
                f"edge {move} does not match the evaluator-derived initial state.",
                event_id=event.event_id,
                phase="expansion",
            )
        # The recorded prior already passed the evaluator-derived check. Keep
        # that exact evidence value when reconstructing resumable state.
        node.edges[move] = _Edge(legal[move], edge.prior)
    node.expanded = True


def _priors_from_logits(logits: dict[str, float], dtype_name: str) -> dict[str, float]:
    ordered = sorted(logits)
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise SemanticReplayError(f"unsupported evaluator dtype {dtype_name!r}.", phase="expansion")
    probabilities = torch.tensor([logits[move] for move in ordered], dtype=dtype).softmax(0).tolist()
    return dict(zip(ordered, probabilities))


def _replay_backups(
    path: list[tuple[_Node, _Edge, _Node]],
    event: SimulationEvent,
    tolerance: ReplayTolerance | None = None,
) -> None:
    if event.leaf.evaluation.value is None:
        raise SemanticReplayError("scalar leaf value is required.", event_id=event.event_id, phase="backup")
    if len(event.backups) != len(path):
        raise SemanticReplayError(
            f"expected {len(path)} backups, got {len(event.backups)}.",
            event_id=event.event_id,
            phase="backup",
        )
    value = event.leaf.evaluation.value
    for index, ((parent, edge, _), recorded) in enumerate(zip(reversed(path), event.backups)):
        value = -value
        perspective = ValuePerspective.ROOT_PLAYER if parent is path[0][0] else ValuePerspective.SIDE_TO_MOVE
        before = _ReferenceMCTS._edge_stat(edge, perspective)
        if (
            recorded.node_id != parent.node_id
            or not _close(recorded.signed_value, value, tolerance)
            or not _same_edge(recorded.before, before, tolerance)
        ):
            raise SemanticReplayError(
                f"backup {index} does not match reconstructed node, value, or pre-state.",
                event_id=event.event_id,
                phase="backup",
            )
        edge.visits += 1
        edge.total_value += value
        after = _ReferenceMCTS._edge_stat(edge, perspective)
        if not _same_edge(recorded.after, after, tolerance):
            raise SemanticReplayError(
                f"backup {index} post-state diverges from the reconstructed update.",
                event_id=event.event_id,
                phase="backup",
            )


def _replay_edge_stats(node: _Node, perspective: ValuePerspective) -> tuple[EdgeStatistics, ...]:
    return tuple(_ReferenceMCTS._edge_stat(node.edges[move], perspective) for move in sorted(node.edges))


def _edge_sequence_discrepancies(
    recorded: tuple[EdgeStatistics, ...] | None,
    replayed: tuple[EdgeStatistics, ...],
    event: SimulationEvent | None,
    phase: str,
    tolerance: ReplayTolerance,
) -> tuple[ReplayDiscrepancy, ...]:
    """Return field-level raw differences without trusting the recorded state."""
    event_id = None if event is None else event.event_id
    if recorded is None:
        return (ReplayDiscrepancy(event_id, phase, "root_statistics", None, replayed),)
    recorded_by_move = {edge.move: edge for edge in recorded}
    replayed_by_move = {edge.move: edge for edge in replayed}
    differences: list[ReplayDiscrepancy] = []
    if len(recorded_by_move) != len(recorded):
        differences.append(ReplayDiscrepancy(event_id, phase, "duplicate_moves", recorded, replayed))
    if recorded_by_move.keys() != replayed_by_move.keys():
        differences.append(
            ReplayDiscrepancy(
                event_id,
                phase,
                "moves",
                tuple(sorted(recorded_by_move)),
                tuple(sorted(replayed_by_move)),
            )
        )
    for move in sorted(recorded_by_move.keys() & replayed_by_move.keys()):
        left = recorded_by_move[move]
        right = replayed_by_move[move]
        for field_name in ("perspective", "prior", "visits", "total_value", "mean_value", "exploration"):
            recorded_value = getattr(left, field_name)
            replayed_value = getattr(right, field_name)
            if not _audit_value_equal(recorded_value, replayed_value, tolerance):
                differences.append(
                    ReplayDiscrepancy(
                        event_id,
                        phase,
                        f"root_statistics.{move}.{field_name}",
                        recorded_value,
                        replayed_value,
                    )
                )
    return tuple(differences)


def _audit_value_equal(left: object, right: object, tolerance: ReplayTolerance) -> bool:
    if isinstance(left, int | float) and not isinstance(left, bool) and isinstance(right, int | float):
        return math.isclose(float(left), float(right), rel_tol=tolerance.relative, abs_tol=tolerance.absolute)
    return left == right


def _same_edge(left: EdgeStatistics, right: EdgeStatistics, tolerance: ReplayTolerance | None = None) -> bool:
    return (
        left.move == right.move
        and left.perspective is right.perspective
        and _close(left.prior, right.prior, tolerance)
        and left.visits == right.visits
        and _close(left.total_value, right.total_value, tolerance)
        and _close(left.mean_value, right.mean_value, tolerance)
        and _close(left.exploration, right.exploration, tolerance)
    )


def _same_retained_edge(left: EdgeStatistics, right: EdgeStatistics) -> bool:
    """Compare replayed root statistics without producer-specific U evidence."""
    return (
        left.move == right.move
        and left.perspective is right.perspective
        and _close(left.prior, right.prior)
        and left.visits == right.visits
        and _close(left.total_value, right.total_value)
        and _close(left.mean_value, right.mean_value)
    )


def _close(left: float | None, right: float | None, tolerance: ReplayTolerance | None = None) -> bool:
    if left is None or right is None:
        return left is right
    policy = tolerance or ReplayTolerance()
    return math.isclose(left, right, rel_tol=policy.relative, abs_tol=policy.absolute)


__all__ = [
    "CounterfactualReplayFormatError",
    "CounterfactualReplayResult",
    "Evaluator",
    "LeafEvaluationReplacement",
    "deserialize_leaf_evaluation_replacement",
    "leaf_evaluation_replacement_digest",
    "ReferenceSearch",
    "ReplayDiscrepancy",
    "ReplayTolerance",
    "RetainedEventReplayCosts",
    "RetainedEventFootprint",
    "RetainedEventPath",
    "RetainedEventReplayPlan",
    "RetainedEventReplayResult",
    "SemanticReplayAudit",
    "SemanticReplayCheckpoint",
    "SemanticReplayError",
    "SemanticReplayResult",
    "audit_search_trace",
    "plan_retained_events",
    "replay_root_events",
    "replay_retained_events",
    "replay_search_trace",
    "serialize_leaf_evaluation_replacement",
]
