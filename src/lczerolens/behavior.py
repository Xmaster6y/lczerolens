"""Typed comparisons for observable evaluator and search behaviour.

This module compares records already produced at the lc0 interoperability and
chess-evidence boundary.  It deliberately does not implement attribution,
probing, hooks, sparse autoencoders, or natural-language explanation methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import chess
import torch
from tensordict import TensorDict

from .board import LczeroBoard
from .counterfactuals import CounterfactualResult
from .move_evidence import VariationEvidence
from .reference_search import replay_root_events
from .search_trace import (
    PositionEvaluation,
    PrincipalVariation,
    RootAction,
    SearchBudget,
    SearchCapability,
    SearchTrace,
    ValuePerspective,
    Wdl,
)


class BehaviorMetric(str, Enum):
    """Stable names for the observable metrics defined by this module."""

    LEGAL_POLICY_LOGIT = "legal_policy_logit"
    LEGAL_POLICY_PROBABILITY = "legal_policy_probability"
    CANDIDATE_RANK = "candidate_rank"
    SELECTED_ACTION = "selected_action"
    VALUE = "value"
    WDL = "wdl"
    MLH = "mlh"
    POLICY_SEARCH_DIVERGENCE = "policy_search_divergence"
    VISIT_AMPLIFICATION = "visit_amplification"
    CANDIDATE_RANK_EVOLUTION = "candidate_rank_evolution"
    Q_EVOLUTION = "q_evolution"
    SELECTED_MOVE_CHANGES = "selected_move_changes"
    DISCOVERY_BUDGET = "discovery_budget"
    PV_STABILITY = "pv_stability"
    EVENT_PATH_DEPTH = "event_path_depth"
    REPLAY_VALIDATION = "replay_validation"


@dataclass(frozen=True)
class MetricDefinition:
    """Complete interpretation contract for one metric."""

    perspective: str
    normalization: str
    aggregation: str
    ties: str
    missing_or_illegal: str
    required_capability: SearchCapability | None = None


_ROOT_DEFINITION = MetricDefinition(
    perspective="each root edge's explicitly stated perspective",
    normalization="visit shares are normalized over exposed root actions with total N > 0",
    aggregation="one budget-labelled root snapshot",
    ties="competition rank; source selection and its declared tie rule are retained",
    missing_or_illegal="unexposed statistics remain None; illegal root actions are rejected by SearchTrace",
    required_capability=SearchCapability.ROOT_ACTION_STATS,
)


METRIC_DEFINITIONS: Mapping[BehaviorMetric, MetricDefinition] = {
    BehaviorMetric.LEGAL_POLICY_LOGIT: MetricDefinition(
        "side to move at the evaluated position",
        "none; raw logits are retained after legal masking",
        "one legal action at one position",
        "exact equality is retained",
        "illegal actions are excluded; policy is required",
    ),
    BehaviorMetric.LEGAL_POLICY_PROBABILITY: MetricDefinition(
        "side to move at the evaluated position",
        "softmax over exactly the legal policy logits",
        "one legal action at one position",
        "exact equality is retained",
        "illegal actions are excluded; policy is required",
    ),
    BehaviorMetric.CANDIDATE_RANK: MetricDefinition(
        "side to move at the evaluated position",
        "competition rank over legal logits",
        "one legal action at one position",
        "equal logits receive equal rank and skip subsequent ranks",
        "illegal actions have no rank; policy is required",
    ),
    BehaviorMetric.SELECTED_ACTION: MetricDefinition(
        "side to move at the evaluated position",
        "maximum legal logit",
        "one position",
        "all exact maxima are retained; UCI lexicographic representative",
        "illegal actions cannot be selected; policy is required",
    ),
    BehaviorMetric.VALUE: MetricDefinition(
        "the EvaluatorBehavior perspective",
        "none; scalar value must be in [-1, 1]",
        "one position",
        "not applicable",
        "missing value head remains None",
    ),
    BehaviorMetric.WDL: MetricDefinition(
        "the EvaluatorBehavior perspective",
        "already-normalized (win, draw, loss) probabilities summing to one",
        "one position",
        "not applicable",
        "missing WDL head remains None",
    ),
    BehaviorMetric.MLH: MetricDefinition(
        "source-reported scalar; no player sign conversion",
        "none",
        "one position",
        "not applicable",
        "missing MLH head remains None",
    ),
    BehaviorMetric.POLICY_SEARCH_DIVERGENCE: _ROOT_DEFINITION,
    BehaviorMetric.VISIT_AMPLIFICATION: MetricDefinition(
        perspective="policy and visits are unsigned root-action weights",
        normalization="normalized visit share divided by evaluator probability conditional on exposed actions",
        aggregation="one root action at one budget-labelled snapshot",
        ties="not applicable",
        missing_or_illegal="None when visits are unavailable, total visits are zero, or conditional probability is zero",
        required_capability=SearchCapability.ROOT_ACTION_STATS,
    ),
    BehaviorMetric.CANDIDATE_RANK_EVOLUTION: MetricDefinition(
        **{**_ROOT_DEFINITION.__dict__, "required_capability": SearchCapability.ROOT_SNAPSHOTS}
    ),
    BehaviorMetric.Q_EVOLUTION: MetricDefinition(
        **{**_ROOT_DEFINITION.__dict__, "required_capability": SearchCapability.ROOT_SNAPSHOTS}
    ),
    BehaviorMetric.SELECTED_MOVE_CHANGES: MetricDefinition(
        **{**_ROOT_DEFINITION.__dict__, "required_capability": SearchCapability.ROOT_SNAPSHOTS}
    ),
    BehaviorMetric.DISCOVERY_BUDGET: MetricDefinition(
        **{**_ROOT_DEFINITION.__dict__, "required_capability": SearchCapability.ROOT_SNAPSHOTS}
    ),
    BehaviorMetric.PV_STABILITY: MetricDefinition(
        perspective="root player; PV moves retain source order",
        normalization="common-prefix length divided by the longer adjacent PV length",
        aggregation="mean over adjacent available PV pairs for each root action",
        ties="not applicable",
        missing_or_illegal="actions with fewer than two exposed PVs receive None",
        required_capability=SearchCapability.ROOT_SNAPSHOTS,
    ),
    BehaviorMetric.EVENT_PATH_DEPTH: MetricDefinition(
        perspective="rooted path depth; no value conversion",
        normalization="none",
        aggregation="maximum and mean across emitted events",
        ties="not applicable",
        missing_or_illegal="refused unless full events are advertised",
        required_capability=SearchCapability.FULL_EVENTS,
    ),
    BehaviorMetric.REPLAY_VALIDATION: MetricDefinition(
        perspective="root player statistics",
        normalization="none",
        aggregation="all emitted event transitions",
        ties="not applicable",
        missing_or_illegal="refused unless replayable events are advertised",
        required_capability=SearchCapability.REPLAYABLE,
    ),
}


@dataclass(frozen=True)
class EvaluatorAction:
    """One legal evaluator action after legal masking and normalization."""

    move: str
    logit: float
    probability: float
    rank: int


@dataclass(frozen=True)
class EvaluatorBehavior:
    """Standardized single-position evaluator behaviour."""

    fen: str
    perspective: ValuePerspective
    actions: tuple[EvaluatorAction, ...]
    selected_move: str
    selection_ties: tuple[str, ...]
    tie_break: str
    evaluation: PositionEvaluation | None = None
    mlh: float | None = None

    def action(self, move: str) -> EvaluatorAction | None:
        """Return a legal action record, or ``None`` for a missing/illegal move."""
        return next((action for action in self.actions if action.move == move), None)


def evaluator_behavior(
    board: LczeroBoard,
    output: TensorDict | Mapping[str, torch.Tensor],
    *,
    perspective: ValuePerspective = ValuePerspective.SIDE_TO_MOVE,
) -> EvaluatorBehavior:
    """Normalize one evaluator output over legal moves with explicit ties.

    ``policy`` is interpreted as raw logits. ``wdl`` is interpreted as the
    already-normalized lc0 ``(win, draw, loss)`` head. Optional scalar heads are
    retained only when present; no head is derived from another head.
    """
    if not isinstance(board, LczeroBoard):
        raise TypeError("Evaluator behaviour requires an LczeroBoard.")
    if not isinstance(perspective, ValuePerspective):
        raise ValueError("perspective must be a ValuePerspective.")
    policy = _single_tensor(output, "policy", required=True)
    if policy.numel() != 1858 or not torch.isfinite(policy).all():
        raise ValueError("policy must contain 1858 finite raw logits for one position.")
    legal_moves = tuple(sorted(board.legal_moves, key=lambda move: move.uci()))
    if not legal_moves:
        raise ValueError("Evaluator behaviour requires a non-terminal position with legal moves.")
    logits = torch.tensor(
        [float(policy[board.encode_move(move, board.turn)].item()) for move in legal_moves], dtype=torch.float64
    )
    probabilities = torch.softmax(logits, dim=0)
    values = logits.tolist()
    ranks = _competition_ranks(values)
    maximum = max(values)
    tied = tuple(move.uci() for move, value in zip(legal_moves, values) if value == maximum)
    actions = tuple(
        EvaluatorAction(move.uci(), logit, float(probability), rank)
        for move, logit, probability, rank in zip(legal_moves, values, probabilities.tolist(), ranks)
    )
    value_tensor = _single_tensor(output, "value")
    wdl_tensor = _single_tensor(output, "wdl")
    mlh_tensor = _single_tensor(output, "mlh")
    value = _optional_scalar(value_tensor, "value", minimum=-1.0, maximum=1.0)
    mlh = _optional_scalar(mlh_tensor, "mlh")
    wdl = None
    if wdl_tensor is not None:
        if wdl_tensor.numel() != 3:
            raise ValueError("wdl must contain exactly (win, draw, loss) for one position.")
        entries = tuple(float(item) for item in wdl_tensor.reshape(-1).tolist())
        wdl = Wdl(*entries, perspective)
    evaluation = (
        PositionEvaluation(perspective, value=value, wdl=wdl) if value is not None or wdl is not None else None
    )
    return EvaluatorBehavior(
        board.fen(),
        perspective,
        actions,
        tied[0],
        tied,
        "UCI lexicographic among exact maximum logits",
        evaluation,
        mlh,
    )


@dataclass(frozen=True)
class SearchActionComparison:
    """Evaluator preference and final root-search statistics for one move."""

    move: str
    evaluator_probability: float | None
    evaluator_rank: int | None
    search_prior: float | None
    visits: int | None
    visit_share: float | None
    visit_probability_delta: float | None
    visit_amplification: float | None
    search_rank: int | None
    search_perspective: ValuePerspective
    mean_value: float | None
    principal_variation: PrincipalVariation | None


@dataclass(frozen=True)
class SearchSnapshotBehavior:
    """Candidate evolution at one source snapshot."""

    sequence: int
    budget: SearchBudget | None
    selected_move: str | None
    ranks: tuple[tuple[str, int | None], ...]
    q_values: tuple[tuple[str, float | None], ...]


@dataclass(frozen=True)
class SearchBehaviorComparison:
    """Root-only evaluator-to-search comparison."""

    source: str
    capability: SearchCapability
    evaluator_selected_move: str
    search_selected_move: str | None
    selection_changed: bool | None
    evaluator_probability_coverage: float
    policy_search_total_variation: float | None
    actions: tuple[SearchActionComparison, ...]
    snapshots: tuple[SearchSnapshotBehavior, ...]
    selected_move_changes: tuple[tuple[int, str, str], ...]
    discovery_budgets: tuple[tuple[str, SearchBudget | None], ...]
    pv_stability: tuple[tuple[str, float | None], ...]

    def action(self, move: str) -> SearchActionComparison | None:
        """Return the comparison for an exposed root action."""
        return next((action for action in self.actions if action.move == move), None)


def compare_evaluator_to_search(evaluator: EvaluatorBehavior, trace: SearchTrace) -> SearchBehaviorComparison:
    """Compare legal evaluator preference with capability-safe root statistics.

    Divergence is total variation between evaluator probabilities and visit
    shares, conditional on the root actions exposed by the search source. The
    separately reported evaluator probability coverage prevents a partial
    source from being mistaken for a complete legal-action comparison.
    """
    trace.require(SearchCapability.ROOT_ACTION_STATS)
    if evaluator.fen != trace.root_fen:
        raise ValueError("Evaluator behaviour and search trace must describe the same root FEN.")
    if evaluator.perspective not in (ValuePerspective.SIDE_TO_MOVE, ValuePerspective.ROOT_PLAYER):
        raise ValueError("Evaluator policy comparison must use side-to-move or root-player perspective.")
    final = trace.snapshots[-1]
    final_actions = final.actions or ()
    evaluator_by_move = {action.move: action for action in evaluator.actions}
    coverage = sum(evaluator_by_move[action.statistics.move].probability for action in final_actions)
    exposed_logits = torch.tensor(
        [evaluator_by_move[action.statistics.move].logit for action in final_actions], dtype=torch.float64
    )
    conditional = {
        action.statistics.move: probability
        for action, probability in zip(final_actions, torch.softmax(exposed_logits, dim=0).tolist())
    }
    visit_total = sum(action.statistics.visits or 0 for action in final_actions)
    visits_available = bool(final_actions) and all(action.statistics.visits is not None for action in final_actions)
    shares = (
        {action.statistics.move: (action.statistics.visits or 0) / visit_total for action in final_actions}
        if visits_available and visit_total > 0
        else {}
    )
    search_ranks = _optional_action_ranks(final_actions)
    comparisons = tuple(
        _search_action_comparison(action, evaluator_by_move[action.statistics.move], conditional, shares, search_ranks)
        for action in final_actions
    )
    divergence = 0.5 * sum(abs(conditional[move] - shares[move]) for move in conditional) if shares else None
    snapshots = tuple(
        _snapshot_behavior(
            snapshot.sequence,
            snapshot.budget,
            snapshot.selection.move if snapshot.selection else None,
            snapshot.actions or (),
        )
        for snapshot in trace.snapshots
    )
    changes = tuple(
        (current.sequence, previous.selected_move, current.selected_move)
        for previous, current in zip(snapshots, snapshots[1:])
        if previous.selected_move is not None
        and current.selected_move is not None
        and previous.selected_move != current.selected_move
    )
    return SearchBehaviorComparison(
        source=trace.provenance.source,
        capability=trace.capability,
        evaluator_selected_move=evaluator.selected_move,
        search_selected_move=final.selection.move if final.selection else None,
        selection_changed=(final.selection.move != evaluator.selected_move) if final.selection else None,
        evaluator_probability_coverage=coverage,
        policy_search_total_variation=divergence,
        actions=comparisons,
        snapshots=snapshots,
        selected_move_changes=changes,
        discovery_budgets=_discovery_budgets(trace),
        pv_stability=_pv_stability(trace),
    )


@dataclass(frozen=True)
class SearchEventBehavior:
    """Metrics which are unavailable from root snapshots alone."""

    event_count: int
    maximum_path_depth: int
    mean_path_depth: float
    expanded_node_count: int
    terminal_leaf_count: int
    replay_validated: bool | None


def compare_search_events(trace: SearchTrace, *, validate_replay: bool = False) -> SearchEventBehavior:
    """Summarize emitted events, refusing unsupported full-event claims."""
    trace.require(SearchCapability.FULL_EVENTS)
    events = trace.events or ()
    depths = [len(event.path) for event in events]
    replay_validated = None
    if validate_replay:
        trace.require(SearchCapability.REPLAYABLE)
        replay_validated = replay_root_events(events) == tuple(
            action.statistics for action in trace.snapshots[-1].actions or ()
        )
    return SearchEventBehavior(
        event_count=len(events),
        maximum_path_depth=max(depths, default=0),
        mean_path_depth=sum(depths) / len(depths) if depths else 0.0,
        expanded_node_count=sum(event.expansion is not None for event in events),
        terminal_leaf_count=sum(event.leaf.terminal for event in events),
        replay_validated=replay_validated,
    )


class ControlKind(str, Enum):
    """First-class counterfactual control relationship."""

    MATCHED = "matched"
    SHUFFLED = "shuffled"
    WRONG_TARGET = "wrong_target"


@dataclass(frozen=True)
class TargetEffect:
    """Behavioural effect for one declared target move."""

    move: str
    original_probability: float | None
    modified_probability: float | None
    probability_delta: float | None
    original_rank: int | None
    modified_rank: int | None
    rank_delta: int | None
    originally_selected: bool
    modified_selected: bool


@dataclass(frozen=True)
class CollateralEffect:
    """Aggregate change outside the explicitly declared targets."""

    moves: tuple[str, ...]
    probability_l1: float
    probability_total_variation: float
    selected_move_changed: bool


@dataclass(frozen=True)
class CounterfactualBehaviorComparison:
    """Target-separated behaviour for a counterfactual or control pair."""

    control_kind: ControlKind
    original: EvaluatorBehavior
    modified: EvaluatorBehavior
    targets: tuple[TargetEffect, ...]
    collateral: CollateralEffect
    counterfactual: CounterfactualResult | None = None
    variation_evidence: tuple[tuple[str, VariationEvidence], ...] = ()


def compare_counterfactual_behavior(
    original: EvaluatorBehavior,
    modified: EvaluatorBehavior,
    target_moves: tuple[str, ...],
    *,
    control_kind: ControlKind = ControlKind.MATCHED,
    counterfactual: CounterfactualResult | None = None,
    variation_evidence: Mapping[str, VariationEvidence] | None = None,
) -> CounterfactualBehaviorComparison:
    """Report declared target effects separately from all collateral effects."""
    if not isinstance(control_kind, ControlKind):
        raise ValueError("control_kind must be a ControlKind.")
    if (
        original.perspective is not modified.perspective
        or chess.Board(original.fen).turn != chess.Board(modified.fen).turn
    ):
        raise ValueError("Counterfactual behaviours must use the same perspective and absolute side to move.")
    if not target_moves or len(target_moves) != len(set(target_moves)):
        raise ValueError("target_moves must be a non-empty tuple of unique UCI moves.")
    for move in target_moves:
        try:
            chess.Move.from_uci(move)
        except ValueError as error:
            raise ValueError(f"Invalid target UCI move: {move!r}.") from error
    if counterfactual is not None and not counterfactual.succeeded:
        raise ValueError("A linked counterfactual result must have succeeded.")
    if counterfactual is not None and (
        counterfactual.original.fen != original.fen
        or counterfactual.modified is None
        or counterfactual.modified.fen != modified.fen
    ):
        raise ValueError("Linked counterfactual positions must match the evaluator behaviours.")
    original_by_move = {action.move: action for action in original.actions}
    modified_by_move = {action.move: action for action in modified.actions}
    targets = tuple(
        _target_effect(move, original, modified, original_by_move.get(move), modified_by_move.get(move))
        for move in target_moves
    )
    collateral_moves = tuple(sorted((original_by_move.keys() | modified_by_move.keys()) - set(target_moves)))
    l1 = sum(
        abs(
            (modified_by_move[move].probability if move in modified_by_move else 0.0)
            - (original_by_move[move].probability if move in original_by_move else 0.0)
        )
        for move in collateral_moves
    )
    evidence = tuple(sorted((variation_evidence or {}).items()))
    unknown_evidence = {move for move, _ in evidence} - (set(target_moves) | set(collateral_moves))
    if unknown_evidence:
        raise ValueError("Variation evidence keys must name a target or collateral move.")
    for move, variation in evidence:
        if not variation.moves or variation.moves[0].uci() != move:
            raise ValueError("Variation evidence must begin with the move named by its key.")
    return CounterfactualBehaviorComparison(
        control_kind,
        original,
        modified,
        targets,
        CollateralEffect(collateral_moves, l1, 0.5 * l1, original.selected_move != modified.selected_move),
        counterfactual,
        evidence,
    )


@dataclass(frozen=True)
class DecisionComparison:
    """Structured checkpoint-D answer without generated explanatory prose."""

    evaluator_candidate: str
    search_candidate: str
    evaluator_candidate_comparison: SearchActionComparison
    search_candidate_comparison: SearchActionComparison
    evaluator_variation: VariationEvidence | None
    search_variation: VariationEvidence | None
    search_source: str
    search_capability: SearchCapability


def compare_search_decision(
    evaluator: EvaluatorBehavior,
    trace: SearchTrace,
    *,
    variation_evidence: Mapping[str, VariationEvidence] | None = None,
) -> DecisionComparison:
    """Link a search-over-evaluator preference change to supplied chess evidence."""
    comparison = compare_evaluator_to_search(evaluator, trace)
    if comparison.search_selected_move is None:
        raise ValueError("Decision comparison requires an exposed search selection.")
    evaluator_action = comparison.action(evaluator.selected_move)
    search_action = comparison.action(comparison.search_selected_move)
    if evaluator_action is None or search_action is None:
        raise ValueError("Both evaluator and search candidates must have exposed root action statistics.")
    evidence = variation_evidence or {}
    for move, variation in evidence.items():
        if variation.initial.fen != evaluator.fen or not variation.moves or variation.moves[0].uci() != move:
            raise ValueError("Decision variation evidence must start at the root and begin with its keyed move.")
    return DecisionComparison(
        evaluator.selected_move,
        comparison.search_selected_move,
        evaluator_action,
        search_action,
        evidence.get(evaluator.selected_move),
        evidence.get(comparison.search_selected_move),
        trace.provenance.source,
        trace.capability,
    )


def _single_tensor(
    output: TensorDict | Mapping[str, torch.Tensor], key: str, *, required: bool = False
) -> torch.Tensor | None:
    value = output.get(key)
    if value is None:
        if required:
            raise ValueError(f"Evaluator output is missing required {key!r} head.")
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Evaluator head {key!r} must be a tensor.")
    if value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value.reshape(-1)


def _optional_scalar(
    value: torch.Tensor | None, name: str, *, minimum: float | None = None, maximum: float | None = None
) -> float | None:
    if value is None:
        return None
    if value.numel() != 1 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be one finite scalar when present.")
    scalar = float(value.item())
    if minimum is not None and scalar < minimum or maximum is not None and scalar > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}].")
    return scalar


def _competition_ranks(values: list[float]) -> tuple[int, ...]:
    return tuple(1 + sum(other > value for other in values) for value in values)


def _optional_action_ranks(actions: tuple[RootAction, ...]) -> Mapping[str, int | None]:
    if not actions or any(action.statistics.visits is None for action in actions):
        return {action.statistics.move: None for action in actions}
    values = [float(action.statistics.visits or 0) for action in actions]
    return {action.statistics.move: rank for action, rank in zip(actions, _competition_ranks(values))}


def _search_action_comparison(
    action: RootAction,
    evaluator: EvaluatorAction,
    conditional: Mapping[str, float],
    shares: Mapping[str, float],
    ranks: Mapping[str, int | None],
) -> SearchActionComparison:
    move = action.statistics.move
    share = shares.get(move)
    probability = conditional[move]
    amplification = share / probability if share is not None and probability > 0 else None
    return SearchActionComparison(
        move,
        evaluator.probability,
        evaluator.rank,
        action.statistics.prior,
        action.statistics.visits,
        share,
        share - probability if share is not None else None,
        amplification,
        ranks[move],
        action.statistics.perspective,
        action.statistics.mean_value,
        action.principal_variation,
    )


def _snapshot_behavior(
    sequence: int,
    budget: SearchBudget | None,
    selected_move: str | None,
    actions: tuple[RootAction, ...],
) -> SearchSnapshotBehavior:
    ranks = _optional_action_ranks(actions)
    return SearchSnapshotBehavior(
        sequence,
        budget,
        selected_move,
        tuple((action.statistics.move, ranks[action.statistics.move]) for action in actions),
        tuple((action.statistics.move, action.statistics.mean_value) for action in actions),
    )


def _discovery_budgets(trace: SearchTrace) -> tuple[tuple[str, SearchBudget | None], ...]:
    if not trace.supports(SearchCapability.ROOT_SNAPSHOTS):
        return ()
    moves = sorted({action.statistics.move for snapshot in trace.snapshots for action in snapshot.actions or ()})
    result = []
    for move in moves:
        budget = next(
            (
                snapshot.budget
                for snapshot in trace.snapshots
                for action in snapshot.actions or ()
                if action.statistics.move == move and (action.statistics.visits or 0) > 0
            ),
            None,
        )
        result.append((move, budget))
    return tuple(result)


def _pv_stability(trace: SearchTrace) -> tuple[tuple[str, float | None], ...]:
    if not trace.supports(SearchCapability.ROOT_SNAPSHOTS):
        return ()
    moves = sorted({action.statistics.move for snapshot in trace.snapshots for action in snapshot.actions or ()})
    result = []
    for move in moves:
        pvs = [
            action.principal_variation.moves
            for snapshot in trace.snapshots
            for action in snapshot.actions or ()
            if action.statistics.move == move and action.principal_variation is not None
        ]
        scores = [_prefix_stability(before, after) for before, after in zip(pvs, pvs[1:])]
        result.append((move, sum(scores) / len(scores) if scores else None))
    return tuple(result)


def _prefix_stability(before: tuple[str, ...], after: tuple[str, ...]) -> float:
    common = 0
    for left, right in zip(before, after):
        if left != right:
            break
        common += 1
    return common / max(len(before), len(after))


def _target_effect(
    move: str,
    original: EvaluatorBehavior,
    modified: EvaluatorBehavior,
    before: EvaluatorAction | None,
    after: EvaluatorAction | None,
) -> TargetEffect:
    before_probability = before.probability if before else None
    after_probability = after.probability if after else None
    return TargetEffect(
        move,
        before_probability,
        after_probability,
        after_probability - before_probability
        if before_probability is not None and after_probability is not None
        else None,
        before.rank if before else None,
        after.rank if after else None,
        after.rank - before.rank if before and after else None,
        original.selected_move == move,
        modified.selected_move == move,
    )


__all__ = [
    "BehaviorMetric",
    "CollateralEffect",
    "ControlKind",
    "CounterfactualBehaviorComparison",
    "DecisionComparison",
    "EvaluatorAction",
    "EvaluatorBehavior",
    "METRIC_DEFINITIONS",
    "MetricDefinition",
    "SearchActionComparison",
    "SearchBehaviorComparison",
    "SearchEventBehavior",
    "SearchSnapshotBehavior",
    "TargetEffect",
    "compare_counterfactual_behavior",
    "compare_evaluator_to_search",
    "compare_search_decision",
    "compare_search_events",
    "evaluator_behavior",
]
