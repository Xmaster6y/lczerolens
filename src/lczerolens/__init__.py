"""Main module for the lczerolens package."""

from importlib.metadata import PackageNotFoundError, version

from ._decision_serialization import DecisionAnalysisFormatError
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
from .evaluator import InputFormat, LczeroEvaluator
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
from .puzzle import (
    Puzzle,
    PuzzleAttempt,
    PuzzleContinuation,
    PuzzleProvenance,
    PuzzleSolution,
    PuzzleStatus,
)
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
    ReplayDiscrepancy,
    ReplayTolerance,
    RetainedEventReplayCosts,
    RetainedEventFootprint,
    RetainedEventPath,
    RetainedEventReplayPlan,
    RetainedEventReplayResult,
    SemanticReplayAudit,
    SemanticReplayCheckpoint,
    SemanticReplayError,
    SemanticReplayResult,
    audit_search_trace,
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
    "LczeroEvaluator",
    "LczeroKeys",
    "LczeroModel",
    "ChessPlayer",
    "CounterfactualConstraints",
    "CounterfactualComparison",
    "CounterfactualPair",
    "CounterfactualValidity",
    "DecisionAnalysis",
    "DecisionAnalysisFormatError",
    "DecisionAction",
    "DecisionActions",
    "Evidence",
    "EvidenceSet",
    "Evaluation",
    "EvaluationBatch",
    "EvaluationRecord",
    "EvaluationRecordFormatError",
    "EvaluationProvenance",
    "FactAnalyzer",
    "ExactMoveEffect",
    "HistoryPolicy",
    "InputFormat",
    "Depth",
    "LczeroSearch",
    "MoveAnalysis",
    "PositionAttribute",
    "PositionIdentity",
    "Puzzle",
    "PuzzleAttempt",
    "PuzzleContinuation",
    "PuzzleProvenance",
    "PuzzleSolution",
    "PuzzleStatus",
    "Nodes",
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
    "audit_search_trace",
    "compare_counterfactual",
    "compare_decision",
    "plan_retained_events",
    "relocate_piece_counterfactual",
    "remove_piece_counterfactual",
    "replay_root_events",
    "replay_retained_events",
    "replay_search_trace",
    "sibling_counterfactual",
]
