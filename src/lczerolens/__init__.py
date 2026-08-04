"""Main module for the lczerolens package."""

from importlib.metadata import PackageNotFoundError, version

from .board import LczeroBoard
from .behavior import (
    BehaviorMetric,
    ControlKind,
    CounterfactualBehaviorComparison,
    DecisionComparison,
    EvaluatorBehavior,
    SearchBehaviorComparison,
    compare_counterfactual_behavior,
    compare_evaluator_to_search,
    compare_search_decision,
    compare_search_events,
    evaluator_behavior,
)
from .counterfactuals import (
    CounterfactualConstraints,
    CounterfactualResult,
    CounterfactualValidity,
    PositionAttribute,
    relocate_piece_counterfactual,
    remove_piece_counterfactual,
    sibling_counterfactual,
)
from .facts import Evidence, EvidenceSet, FactAnalyzer
from .lc0_adapter import Lc0ProcessAdapter, Lc0RootSnapshotParser, Lc0SearchRequest
from .model import LczeroModel
from .move_evidence import MoveDelta, VariationEvidence, analyze_move_delta, analyze_variation
from .reference_search import (
    ReferenceMCTS,
    RetainedEventReplayCosts,
    RetainedEventReplayPlan,
    RetainedEventReplayResult,
    SemanticReplayError,
    SemanticReplayResult,
    plan_retained_events,
    replay_retained_events,
    replay_root_events,
    replay_search_trace,
)

try:
    __version__ = version("lczerolens")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "LczeroBoard",
    "LczeroModel",
    "BehaviorMetric",
    "ControlKind",
    "CounterfactualConstraints",
    "CounterfactualBehaviorComparison",
    "CounterfactualResult",
    "CounterfactualValidity",
    "DecisionComparison",
    "Evidence",
    "EvidenceSet",
    "EvaluatorBehavior",
    "FactAnalyzer",
    "Lc0ProcessAdapter",
    "Lc0RootSnapshotParser",
    "Lc0SearchRequest",
    "MoveDelta",
    "PositionAttribute",
    "ReferenceMCTS",
    "RetainedEventReplayCosts",
    "RetainedEventReplayPlan",
    "RetainedEventReplayResult",
    "SemanticReplayError",
    "SemanticReplayResult",
    "SearchBehaviorComparison",
    "VariationEvidence",
    "analyze_move_delta",
    "analyze_variation",
    "compare_counterfactual_behavior",
    "compare_evaluator_to_search",
    "compare_search_decision",
    "compare_search_events",
    "evaluator_behavior",
    "plan_retained_events",
    "relocate_piece_counterfactual",
    "remove_piece_counterfactual",
    "replay_root_events",
    "replay_retained_events",
    "replay_search_trace",
    "sibling_counterfactual",
]
