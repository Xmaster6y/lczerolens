"""Concrete evaluator, search, and counterfactual decision comparisons."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

from .counterfactuals import CounterfactualPair
from .evaluation import (
    ActionEvaluationRecord,
    Evaluation,
    EvaluationRecord,
    ScalarEvaluationRecord,
)
from .moves import LineAnalysis, analyze_line
from .search.result import SearchResult
from .search.trace import PrincipalVariation, RootAction, SearchCapability, ValuePerspective

if TYPE_CHECKING:
    from .evaluator import LczeroEvaluator


@dataclass(frozen=True)
class DecisionAction:
    """Policy and search observations for one legal root action."""

    move: str
    policy_logit: float
    policy_probability: float
    policy_rank: int
    search_prior: float | None
    search_visits: int | None
    search_visit_share: float | None
    search_rank: int | None
    search_value: float | None
    search_value_perspective: ValuePerspective | None
    principal_variation: PrincipalVariation | None
    line: LineAnalysis | None
    selected_by_policy: bool
    selected_by_search: bool


class DecisionActions(Mapping[str, DecisionAction]):
    """Immutable move-keyed decision actions in canonical UCI order."""

    def __init__(self, actions: Sequence[DecisionAction]):
        self._actions = tuple(actions)
        self._by_move = {action.move: action for action in self._actions}
        if len(self._actions) != len(self._by_move):
            raise ValueError("Decision actions must have unique moves.")
        if tuple(self._by_move) != tuple(sorted(self._by_move)):
            raise ValueError("Decision actions must use canonical UCI order.")

    def __getitem__(self, move: str | chess.Move) -> DecisionAction:
        key = move if isinstance(move, str) else move.uci()
        return self._by_move[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_move)

    def __len__(self) -> int:
        return len(self._actions)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DecisionActions) and self._actions == other._actions


@dataclass(frozen=True)
class DecisionAnalysis:
    """Evaluator and search evidence joined by position and action identity."""

    evaluation: EvaluationRecord
    search: SearchResult
    policy_move: str
    search_move: str
    changed: bool
    actions: DecisionActions
    counterfactuals: tuple[CounterfactualComparison, ...] = ()

    def __post_init__(self) -> None:
        if self.evaluation.position.fen != self.search.trace.root_fen:
            raise ValueError("Decision evaluation and search must describe the same root position.")
        if self.policy_move not in self.actions or self.search_move not in self.actions:
            raise ValueError("Policy and search selections must be present in decision actions.")
        if self.changed is not (self.policy_move != self.search_move):
            raise ValueError("Decision changed status must match the selected moves.")
        if any(not isinstance(item, CounterfactualComparison) for item in self.counterfactuals):
            raise ValueError("Decision counterfactuals must be CounterfactualComparison values.")
        if any(
            item.factual_evaluation.provenance != self.evaluation.provenance
            or item.alternative_evaluation.provenance != self.evaluation.provenance
            for item in self.counterfactuals
        ):
            raise ValueError("Decision counterfactuals must use the decision evaluator provenance.")

    @property
    def search_source(self) -> str:
        return self.search.trace.provenance.source

    @property
    def search_capability(self) -> SearchCapability:
        return self.search.trace.capability


@dataclass(frozen=True)
class CounterfactualActionChange:
    """Evaluator-policy change for one move in the union of both positions."""

    move: str
    factual_probability: float | None
    alternative_probability: float | None
    probability_delta: float | None
    factual_rank: int | None
    alternative_rank: int | None


@dataclass(frozen=True)
class CounterfactualPolicyChange:
    """Selection and distribution change between factual and alternative evaluations."""

    factual_move: str | None
    alternative_move: str | None
    selected_move_changed: bool
    total_variation: float
    actions: tuple[CounterfactualActionChange, ...]

    def __getitem__(self, move: str | chess.Move) -> CounterfactualActionChange:
        key = move if isinstance(move, str) else move.uci()
        return next(action for action in self.actions if action.move == key)


@dataclass(frozen=True)
class CounterfactualValueChange:
    """Scalar evaluator change without deriving or changing its perspective."""

    factual: ScalarEvaluationRecord | None
    alternative: ScalarEvaluationRecord | None
    delta: float | None


@dataclass(frozen=True)
class CounterfactualComparison:
    """Model observations for an already-constructed counterfactual pair."""

    pair: CounterfactualPair
    factual_evaluation: EvaluationRecord
    alternative_evaluation: EvaluationRecord
    policy_change: CounterfactualPolicyChange
    value_change: CounterfactualValueChange


def compare_decision(
    evaluation: Evaluation | EvaluationRecord,
    search: SearchResult,
    *,
    line_analyses: Mapping[str, LineAnalysis] | None = None,
    counterfactuals: Sequence[CounterfactualComparison] = (),
) -> DecisionAnalysis:
    """Compare evaluator and search selections while retaining both producers' evidence."""
    record = evaluation.record() if isinstance(evaluation, Evaluation) else evaluation
    if not isinstance(record, EvaluationRecord):
        raise TypeError("evaluation must be an Evaluation or EvaluationRecord.")
    if not isinstance(search, SearchResult):
        raise TypeError("search must be a SearchResult.")
    trace = search.trace
    if record.position.fen != trace.root_fen:
        raise ValueError("Evaluation and search must describe the same root position.")
    policy_move = _selected_policy_move(record)
    final = trace.snapshots[-1]
    search_move = search.move.uci()
    root_actions = {action.statistics.move: action for action in final.actions or ()}
    ranks = _search_ranks(tuple(root_actions.values()))
    visits = [action.statistics.visits for action in root_actions.values()]
    visit_total = (
        sum(value for value in visits if value is not None) if visits and all(v is not None for v in visits) else None
    )
    supplied = dict(line_analyses or {})
    board = record.position.board()
    for move, line in supplied.items():
        if move not in {action.move for action in record.policy}:
            raise ValueError("Line-analysis keys must name legal evaluated root moves.")
        if line.initial_position.fen != record.position.fen or not line.moves or line.moves[0].uci() != move:
            raise ValueError("Decision line analysis must start at the root and begin with its keyed move.")
    actions = DecisionActions(
        tuple(
            _decision_action(
                action,
                root_actions.get(action.move),
                ranks.get(action.move),
                visit_total,
                supplied.get(action.move),
                board,
                action.move in (policy_move, search_move),
                policy_move,
                search_move,
            )
            for action in record.policy
        )
    )
    return DecisionAnalysis(
        record,
        search,
        policy_move,
        search_move,
        policy_move != search_move,
        actions,
        tuple(counterfactuals),
    )


def compare_counterfactual(pair: CounterfactualPair, evaluator: LczeroEvaluator) -> CounterfactualComparison:
    """Evaluate a successful pair and report model changes separately from pair validity."""
    if not isinstance(pair, CounterfactualPair):
        raise TypeError("pair must be a CounterfactualPair.")
    if not pair.succeeded or pair.alternative is None:
        raise ValueError("Counterfactual comparison requires a successfully constructed pair.")
    if not hasattr(evaluator, "evaluate"):
        raise TypeError("evaluator must provide the LczeroEvaluator evaluation interface.")
    evaluated = evaluator.evaluate((pair.factual.board(), pair.alternative.board()))
    if not isinstance(evaluated, Sequence) or len(evaluated) != 2:
        raise TypeError("Counterfactual evaluation must return exactly two Evaluation views.")
    factual, alternative = evaluated
    if not isinstance(factual, Evaluation) or not isinstance(alternative, Evaluation):
        raise TypeError("Counterfactual evaluation must return Evaluation views.")
    factual_record = factual.record()
    alternative_record = alternative.record()
    return CounterfactualComparison(
        pair,
        factual_record,
        alternative_record,
        _policy_change(factual_record, alternative_record),
        _value_change(factual_record.value, alternative_record.value),
    )


def _selected_policy_move(record: EvaluationRecord) -> str:
    selected = min((action.move for action in record.policy if action.rank == 1), default=None)
    if selected is None:
        raise ValueError("Decision analysis requires a non-terminal evaluation policy.")
    return selected


def _search_ranks(actions: tuple[RootAction, ...]) -> dict[str, int | None]:
    if not actions or any(action.statistics.visits is None for action in actions):
        return {action.statistics.move: None for action in actions}
    visits = [action.statistics.visits or 0 for action in actions]
    return {
        action.statistics.move: 1 + sum(other > visits[index] for other in visits)
        for index, action in enumerate(actions)
    }


def _decision_action(
    policy: ActionEvaluationRecord,
    search: RootAction | None,
    search_rank: int | None,
    visit_total: int | None,
    supplied_line: LineAnalysis | None,
    board: chess.Board,
    needs_line: bool,
    policy_move: str,
    search_move: str,
) -> DecisionAction:
    statistics = search.statistics if search is not None else None
    pv = search.principal_variation if search is not None else None
    line = supplied_line
    if line is None and pv is not None:
        line = analyze_line(board, pv.moves)
    elif line is None and needs_line:
        line = analyze_line(board, (policy.move,))
    visits = statistics.visits if statistics is not None else None
    return DecisionAction(
        move=policy.move,
        policy_logit=policy.logit,
        policy_probability=policy.probability,
        policy_rank=policy.rank,
        search_prior=statistics.prior if statistics is not None else None,
        search_visits=visits,
        search_visit_share=visits / visit_total if visits is not None and visit_total else None,
        search_rank=search_rank,
        search_value=statistics.mean_value if statistics is not None else None,
        search_value_perspective=statistics.perspective if statistics is not None else None,
        principal_variation=pv,
        line=line,
        selected_by_policy=policy.move == policy_move,
        selected_by_search=policy.move == search_move,
    )


def _policy_change(factual: EvaluationRecord, alternative: EvaluationRecord) -> CounterfactualPolicyChange:
    factual_actions = {action.move: action for action in factual.policy}
    alternative_actions = {action.move: action for action in alternative.policy}
    moves = tuple(sorted(factual_actions.keys() | alternative_actions.keys()))
    changes = tuple(
        CounterfactualActionChange(
            move,
            factual_actions[move].probability if move in factual_actions else None,
            alternative_actions[move].probability if move in alternative_actions else None,
            (
                alternative_actions[move].probability - factual_actions[move].probability
                if move in factual_actions and move in alternative_actions
                else None
            ),
            factual_actions[move].rank if move in factual_actions else None,
            alternative_actions[move].rank if move in alternative_actions else None,
        )
        for move in moves
    )
    total_variation = 0.5 * sum(
        abs(
            (alternative_actions[move].probability if move in alternative_actions else 0.0)
            - (factual_actions[move].probability if move in factual_actions else 0.0)
        )
        for move in moves
    )
    factual_move = _optional_selected_policy_move(factual)
    alternative_move = _optional_selected_policy_move(alternative)
    return CounterfactualPolicyChange(
        factual_move,
        alternative_move,
        factual_move != alternative_move,
        total_variation,
        changes,
    )


def _optional_selected_policy_move(record: EvaluationRecord) -> str | None:
    return min((action.move for action in record.policy if action.rank == 1), default=None)


def _value_change(
    factual: ScalarEvaluationRecord | None,
    alternative: ScalarEvaluationRecord | None,
) -> CounterfactualValueChange:
    delta = None
    if factual is not None and alternative is not None and factual.perspective is alternative.perspective:
        delta = alternative.value - factual.value
    return CounterfactualValueChange(factual, alternative, delta)


__all__ = [
    "CounterfactualActionChange",
    "CounterfactualComparison",
    "CounterfactualPolicyChange",
    "CounterfactualValueChange",
    "DecisionAction",
    "DecisionActions",
    "DecisionAnalysis",
    "compare_counterfactual",
    "compare_decision",
]
