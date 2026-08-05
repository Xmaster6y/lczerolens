"""Main module for the lczerolens package."""

from importlib.metadata import PackageNotFoundError, version

from .board import LczeroBoard
from .behavior import (
    BehaviorMetric,
    ControlKind,
    CounterfactualBehaviorComparison,
    EvaluatorBehavior,
    SearchBehaviorComparison,
    compare_counterfactual_behavior,
    compare_evaluator_to_search,
    compare_search_events,
    evaluator_behavior,
)
from .counterfactuals import (
    CounterfactualConstraints,
    CounterfactualPair,
    CounterfactualValidity,
    PositionAttribute,
    relocate_piece_counterfactual,
    remove_piece_counterfactual,
    sibling_counterfactual,
)
from .decision import (
    CounterfactualComparison,
    DecisionAction,
    DecisionActions,
    DecisionAnalysis,
    compare_counterfactual,
    compare_decision,
)
from .facts import Evidence, EvidenceSet, FactAnalyzer
from .evaluation import Evaluation, EvaluationBatch, EvaluationRecord
from .evaluator import LczeroEvaluator
from .model import LczeroModel
from .moves import (
    ExactMoveEffect,
    HistoryPolicy,
    LineAnalysis,
    LineAnalysisError,
    LineFailureReason,
    LineIntent,
    LineRole,
    LineTerminal,
    MoveAnalysis,
    analyze_line,
    analyze_move,
)
from .provenance import ChessPlayer, EvaluationProvenance, PositionIdentity
from .search import (
    Depth,
    LczeroSearch,
    Nodes,
    ReferenceSearch,
    SearchAction,
    SearchEvidenceUnavailable,
    SearchLimit,
    SearchResult,
    SearchRoot,
    Simulations,
    Time,
    Visits,
)
from .search.reference import (
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
from .serialization import EvaluationRecordFormatError
from .schema import LczeroKeys

try:
    __version__ = version("lczerolens")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "LczeroBoard",
    "LczeroEvaluator",
    "LczeroKeys",
    "LczeroModel",
    "BehaviorMetric",
    "ChessPlayer",
    "ControlKind",
    "CounterfactualConstraints",
    "CounterfactualComparison",
    "CounterfactualBehaviorComparison",
    "CounterfactualPair",
    "CounterfactualValidity",
    "DecisionAnalysis",
    "DecisionAction",
    "DecisionActions",
    "Evidence",
    "EvidenceSet",
    "Evaluation",
    "EvaluationBatch",
    "EvaluationRecord",
    "EvaluationRecordFormatError",
    "EvaluationProvenance",
    "EvaluatorBehavior",
    "FactAnalyzer",
    "ExactMoveEffect",
    "HistoryPolicy",
    "Depth",
    "LczeroSearch",
    "MoveAnalysis",
    "PositionAttribute",
    "PositionIdentity",
    "Nodes",
    "ReferenceSearch",
    "RetainedEventReplayCosts",
    "RetainedEventReplayPlan",
    "RetainedEventReplayResult",
    "SemanticReplayError",
    "SemanticReplayResult",
    "SearchBehaviorComparison",
    "SearchAction",
    "SearchEvidenceUnavailable",
    "SearchLimit",
    "SearchResult",
    "SearchRoot",
    "Simulations",
    "Time",
    "Visits",
    "LineAnalysis",
    "LineAnalysisError",
    "LineFailureReason",
    "LineIntent",
    "LineRole",
    "LineTerminal",
    "analyze_move",
    "analyze_line",
    "compare_counterfactual_behavior",
    "compare_counterfactual",
    "compare_evaluator_to_search",
    "compare_decision",
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
