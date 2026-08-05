"""Canonical persistence for complete decision-analysis evidence."""

from __future__ import annotations

from functools import cache
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from os import PathLike
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import chess

from lczerolens.counterfactuals import (
    AttributeChange,
    AttributeVerification,
    ConstraintRelation,
    CounterfactualFailure,
    CounterfactualFailureReason,
    CounterfactualPair,
    CounterfactualPosition,
    CounterfactualValidity,
    FactVerification,
    HistoryGuarantee,
    PositionAttribute,
    RelocatePieceOperator,
    RemovePieceOperator,
    SiblingMoveOperator,
)
from lczerolens.decision import (
    CounterfactualComparison,
    DecisionAnalysis,
    _policy_change,
    _value_change,
    compare_decision,
)
from lczerolens.facts import (
    AnalyzerProvenance,
    AttacksDefendersValue,
    ChessSide,
    Evidence,
    EvidenceSet,
    FactKind,
    FactPerspective,
    FactScope,
    Guarantee,
    HistoryRequirement,
    MoveSubject,
    PieceSubject,
    RegionSubject,
    SideSubject,
    SquareSubject,
    SupportingPiece,
    UndefinedReason,
)
from lczerolens.moves import (
    EvidenceTransition,
    EvidenceTransitionKind,
    ExactMoveEffect,
    HistoryPolicy,
    LineAnalysis,
    LineIntent,
    LineRole,
    LineTerminal,
    MoveAnalysis,
    MoveEvidence,
    PositionEvidence,
)
from lczerolens.provenance import PositionIdentity
from lczerolens.search.result import SearchResult
from lczerolens.search.trace import deserialize_search_trace, serialize_search_trace
from lczerolens.serialization import deserialize_evaluation_record, serialize_evaluation_record


class DecisionAnalysisFormatError(ValueError):
    """Raised when bytes do not satisfy the canonical decision format."""


_FORMAT = "lczerolens.decision-analysis"
_FORMAT_VERSION = 1

_RECORD_TYPES = {
    record_type.__name__: record_type
    for record_type in (
        AnalyzerProvenance,
        AttacksDefendersValue,
        AttributeChange,
        AttributeVerification,
        CounterfactualFailure,
        CounterfactualPair,
        CounterfactualPosition,
        Evidence,
        EvidenceSet,
        EvidenceTransition,
        FactVerification,
        HistoryGuarantee,
        LineAnalysis,
        LineIntent,
        LineTerminal,
        MoveAnalysis,
        MoveEvidence,
        MoveSubject,
        PieceSubject,
        PositionEvidence,
        PositionIdentity,
        RegionSubject,
        RelocatePieceOperator,
        RemovePieceOperator,
        SideSubject,
        SiblingMoveOperator,
        SquareSubject,
        SupportingPiece,
    )
}
_RECORD_NAMES = {record_type: name for name, record_type in _RECORD_TYPES.items()}
_ENUM_TYPES = {
    enum_type.__name__: enum_type
    for enum_type in (
        chess.Status,
        chess.Termination,
        ChessSide,
        ConstraintRelation,
        CounterfactualFailureReason,
        CounterfactualValidity,
        EvidenceTransitionKind,
        ExactMoveEffect,
        FactKind,
        FactPerspective,
        FactScope,
        Guarantee,
        HistoryPolicy,
        HistoryRequirement,
        LineRole,
        PositionAttribute,
        UndefinedReason,
    )
}
_ENUM_NAMES = {enum_type: name for name, enum_type in _ENUM_TYPES.items()}


def serialize_decision_analysis(decision: DecisionAnalysis) -> bytes:
    """Return canonical versioned JSON bytes for a complete decision analysis."""
    if not isinstance(decision, DecisionAnalysis):
        raise TypeError("serialize_decision_analysis expects a DecisionAnalysis.")
    lines = [
        {"line": _canonical_evidence(action.line), "move": move}
        for move, action in decision.actions.items()
        if action.line is not None
    ]
    counterfactuals = [
        {
            "alternative_evaluation": _json_object(serialize_evaluation_record(item.alternative_evaluation)),
            "factual_evaluation": _json_object(serialize_evaluation_record(item.factual_evaluation)),
            "pair": _canonical_evidence(item.pair),
        }
        for item in decision.counterfactuals
    ]
    envelope = {
        "counterfactuals": counterfactuals,
        "evaluation": _json_object(serialize_evaluation_record(decision.evaluation)),
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "lines": lines,
        "search_trace": _json_object(serialize_search_trace(decision.search.trace)),
    }
    return _canonical_json(envelope)


def deserialize_decision_analysis(data: bytes) -> DecisionAnalysis:
    """Restore complete decision evidence, rejecting malformed or noncanonical bytes."""
    if not isinstance(data, bytes):
        raise TypeError("deserialize_decision_analysis expects bytes.")
    try:
        envelope = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except DecisionAnalysisFormatError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DecisionAnalysisFormatError(f"Invalid decision-analysis JSON: {error}") from error
    _require_fields(
        envelope,
        {"counterfactuals", "evaluation", "format", "format_version", "lines", "search_trace"},
        "Decision envelope",
    )
    if envelope["format"] != _FORMAT:
        raise DecisionAnalysisFormatError(f"Unsupported decision-analysis format {envelope['format']!r}.")
    if isinstance(envelope["format_version"], bool) or envelope["format_version"] != _FORMAT_VERSION:
        raise DecisionAnalysisFormatError(
            f"Unsupported decision-analysis format version {envelope['format_version']!r}."
        )
    try:
        evaluation = deserialize_evaluation_record(_canonical_json(envelope["evaluation"]))
        trace = deserialize_search_trace(_canonical_json(envelope["search_trace"]))
        lines = _decode_lines(envelope["lines"])
        counterfactuals = _decode_counterfactuals(envelope["counterfactuals"])
        decision = compare_decision(
            evaluation,
            SearchResult.from_trace(trace),
            line_analyses=lines,
            counterfactuals=counterfactuals,
        )
    except DecisionAnalysisFormatError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise DecisionAnalysisFormatError(f"Invalid DecisionAnalysis: {error}") from error
    if serialize_decision_analysis(decision) != data:
        raise DecisionAnalysisFormatError("Decision-analysis bytes are valid JSON but are not canonical.")
    return decision


def decision_analysis_digest(decision: DecisionAnalysis) -> str:
    """Return the lowercase SHA-256 digest of canonical decision bytes."""
    return hashlib.sha256(serialize_decision_analysis(decision)).hexdigest()


def save_decision_analysis(decision: DecisionAnalysis, path: str | PathLike[str]) -> None:
    """Write canonical decision bytes to ``path``."""
    Path(path).write_bytes(serialize_decision_analysis(decision))


def load_decision_analysis(path: str | PathLike[str]) -> DecisionAnalysis:
    """Load canonical decision bytes from ``path``."""
    return deserialize_decision_analysis(Path(path).read_bytes())


def _decode_lines(value: Any) -> dict[str, LineAnalysis]:
    if not isinstance(value, list):
        raise DecisionAnalysisFormatError("Decision lines must be an array.")
    result: dict[str, LineAnalysis] = {}
    for entry in value:
        _require_fields(entry, {"line", "move"}, "Decision line")
        move = entry["move"]
        if not isinstance(move, str):
            raise DecisionAnalysisFormatError("Decision line moves must be strings.")
        line = _decode_evidence(entry["line"])
        if not isinstance(line, LineAnalysis):
            raise DecisionAnalysisFormatError("Decision line entries must contain LineAnalysis records.")
        if move in result:
            raise DecisionAnalysisFormatError(f"Duplicate decision line move {move!r}.")
        result[move] = line
    if tuple(result) != tuple(sorted(result)):
        raise DecisionAnalysisFormatError("Decision lines must use canonical UCI order.")
    return result


def _decode_counterfactuals(value: Any) -> tuple[CounterfactualComparison, ...]:
    if not isinstance(value, list):
        raise DecisionAnalysisFormatError("Decision counterfactuals must be an array.")
    comparisons = []
    for entry in value:
        _require_fields(
            entry,
            {"alternative_evaluation", "factual_evaluation", "pair"},
            "Decision counterfactual",
        )
        pair = _decode_evidence(entry["pair"])
        if not isinstance(pair, CounterfactualPair):
            raise DecisionAnalysisFormatError("Decision counterfactuals must contain CounterfactualPair records.")
        factual = deserialize_evaluation_record(_canonical_json(entry["factual_evaluation"]))
        alternative = deserialize_evaluation_record(_canonical_json(entry["alternative_evaluation"]))
        comparisons.append(
            CounterfactualComparison(
                pair,
                factual,
                alternative,
                _policy_change(factual, alternative),
                _value_change(factual.value, alternative.value),
            )
        )
    return tuple(comparisons)


def _canonical_evidence(value: Any) -> Any:
    if isinstance(value, chess.Move):
        return {"$chess_move": value.uci()}
    if isinstance(value, chess.Piece):
        return {"$chess_piece": {"color": value.color, "piece_type": value.piece_type}}
    if isinstance(value, Enum):
        enum_name = _ENUM_NAMES.get(type(value))
        if enum_name is None:
            raise DecisionAnalysisFormatError(f"Unsupported decision enum {type(value).__name__!r}.")
        return {"$enum": enum_name, "value": value.value}
    if is_dataclass(value) and not isinstance(value, type):
        type_name = _RECORD_NAMES.get(type(value))
        if type_name is None:
            raise DecisionAnalysisFormatError(f"Unsupported decision record {type(value).__name__!r}.")
        return {
            "$type": type_name,
            **{item.name: _canonical_evidence(getattr(value, item.name)) for item in fields(value)},
        }
    if isinstance(value, tuple):
        return [_canonical_evidence(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DecisionAnalysisFormatError("Canonical decision analyses cannot contain non-finite floats.")
        return int(value) if value == 0 or value.is_integer() else value
    if value is None or isinstance(value, str | int | bool):
        return value
    raise DecisionAnalysisFormatError(f"Unsupported decision value {type(value).__name__!r}.")


def _decode_evidence(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode_evidence(item) for item in value)
    if not isinstance(value, dict):
        return value
    tags = set(value) & {"$chess_move", "$chess_piece", "$enum", "$type"}
    if len(tags) != 1:
        raise DecisionAnalysisFormatError("Nested decision objects require exactly one supported type tag.")
    tag = next(iter(tags))
    if tag == "$chess_move":
        _require_fields(value, {tag}, "Chess move")
        try:
            return chess.Move.from_uci(value[tag])
        except (AttributeError, TypeError, ValueError) as error:
            raise DecisionAnalysisFormatError("Invalid canonical chess move.") from error
    if tag == "$chess_piece":
        _require_fields(value, {tag}, "Chess piece")
        piece = value[tag]
        _require_fields(piece, {"color", "piece_type"}, "Chess piece value")
        if not isinstance(piece["color"], bool) or (
            isinstance(piece["piece_type"], bool)
            or not isinstance(piece["piece_type"], int)
            or piece["piece_type"] not in chess.PIECE_TYPES
        ):
            raise DecisionAnalysisFormatError("Invalid canonical chess piece.")
        return chess.Piece(piece["piece_type"], piece["color"])
    if tag == "$enum":
        _require_fields(value, {"$enum", "value"}, "Decision enum")
        enum_name = value["$enum"]
        if not isinstance(enum_name, str) or enum_name not in _ENUM_TYPES:
            raise DecisionAnalysisFormatError(f"Unknown decision enum {enum_name!r}.")
        try:
            return _ENUM_TYPES[enum_name](value["value"])
        except (TypeError, ValueError) as error:
            raise DecisionAnalysisFormatError(f"Invalid {enum_name} value {value['value']!r}.") from error
    type_name = value["$type"]
    if not isinstance(type_name, str) or type_name not in _RECORD_TYPES:
        raise DecisionAnalysisFormatError(f"Unknown decision record type {type_name!r}.")
    record_type = _RECORD_TYPES[type_name]
    expected = {item.name for item in fields(record_type)}
    actual = set(value) - {"$type"}
    if actual != expected:
        raise DecisionAnalysisFormatError(
            f"Invalid {type_name} fields: missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}."
        )
    decoded = {name: _decode_evidence(value[name]) for name in expected}
    annotations = _record_type_hints(record_type)
    for name in expected:
        try:
            decoded[name] = _coerce_annotation(decoded[name], annotations[name])
        except TypeError as error:
            raise DecisionAnalysisFormatError(
                f"Invalid {type_name}.{name}: value does not match its declared field type."
            ) from error
    try:
        return record_type(**decoded)
    except (AttributeError, TypeError, ValueError) as error:
        raise DecisionAnalysisFormatError(f"Invalid {type_name} record: {error}") from error


@cache
def _record_type_hints(record_type: type[Any]) -> dict[str, Any]:
    return get_type_hints(record_type)


def _coerce_annotation(value: Any, annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        for option in get_args(annotation):
            try:
                return _coerce_annotation(value, option)
            except TypeError:
                continue
        raise TypeError
    if origin is tuple:
        if not isinstance(value, tuple):
            raise TypeError
        arguments = get_args(annotation)
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_coerce_annotation(item, arguments[0]) for item in value)
        if len(value) != len(arguments):
            raise TypeError
        return tuple(_coerce_annotation(item, item_type) for item, item_type in zip(value, arguments))
    if annotation in (Any, object):
        return value
    if annotation is type(None):
        if value is not None:
            raise TypeError
        return None
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError
        return value
    if isinstance(annotation, type) and isinstance(value, annotation):
        return value
    raise TypeError


def _json_object(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise DecisionAnalysisFormatError(f"Decision analysis is not canonical JSON: {error}") from error


def _require_fields(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise DecisionAnalysisFormatError(f"{label} must be an object.")
    actual = set(value)
    if actual != expected:
        raise DecisionAnalysisFormatError(
            f"Invalid {label} fields: missing={sorted(expected - actual)!r}, unexpected={sorted(actual - expected)!r}."
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DecisionAnalysisFormatError(f"Duplicate JSON field {key!r}.")
        value[key] = item
    return value


__all__ = [
    "DecisionAnalysisFormatError",
    "decision_analysis_digest",
    "deserialize_decision_analysis",
    "load_decision_analysis",
    "save_decision_analysis",
    "serialize_decision_analysis",
]
