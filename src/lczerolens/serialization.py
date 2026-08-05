"""Canonical versioned persistence for lczerolens evidence records."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from lczerolens.evaluation import (
    ActionEvaluationRecord,
    EvaluationRecord,
    ScalarEvaluationRecord,
    ValueOrigin,
    WdlEvaluationRecord,
)
from lczerolens.provenance import ChessPlayer, EvaluationProvenance, PositionIdentity


class EvaluationRecordFormatError(ValueError):
    """Raised when bytes do not satisfy the canonical evaluation format."""


_FORMAT = "lczerolens.evaluation"
_FORMAT_VERSION = 1


def serialize_evaluation_record(record: EvaluationRecord) -> bytes:
    """Return canonical UTF-8 JSON bytes for an evaluation record."""
    if not isinstance(record, EvaluationRecord):
        raise TypeError("serialize_evaluation_record expects an EvaluationRecord.")
    envelope = {
        "format": _FORMAT,
        "format_version": _FORMAT_VERSION,
        "record": _record_data(record),
    }
    try:
        return json.dumps(
            envelope,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except UnicodeEncodeError as error:
        raise EvaluationRecordFormatError("Canonical evaluation records require valid UTF-8 strings.") from error


def deserialize_evaluation_record(data: bytes) -> EvaluationRecord:
    """Restore an evaluation record, rejecting malformed or noncanonical bytes."""
    if not isinstance(data, bytes):
        raise TypeError("deserialize_evaluation_record expects bytes.")
    try:
        envelope = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except EvaluationRecordFormatError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationRecordFormatError(f"Invalid evaluation-record JSON: {error}") from error
    _require_fields(envelope, {"format", "format_version", "record"}, "Evaluation envelope")
    if envelope["format"] != _FORMAT:
        raise EvaluationRecordFormatError(f"Unsupported evaluation-record format {envelope['format']!r}.")
    if isinstance(envelope["format_version"], bool) or envelope["format_version"] != _FORMAT_VERSION:
        raise EvaluationRecordFormatError(
            f"Unsupported evaluation-record format version {envelope['format_version']!r}."
        )
    record = _decode_record(envelope["record"])
    if serialize_evaluation_record(record) != data:
        raise EvaluationRecordFormatError("Evaluation-record bytes are valid JSON but are not canonical.")
    return record


def evaluation_record_digest(record: EvaluationRecord) -> str:
    """Return the lowercase SHA-256 digest of canonical record bytes."""
    return hashlib.sha256(serialize_evaluation_record(record)).hexdigest()


def _number(value: float | int) -> int | float:
    if isinstance(value, bool):
        raise EvaluationRecordFormatError("Canonical evaluation records require numeric evidence values.")
    if isinstance(value, int):
        return value
    numeric = float(value)
    if not math.isfinite(numeric):
        raise EvaluationRecordFormatError("Canonical evaluation records cannot contain non-finite floats.")
    if numeric == 0 or numeric.is_integer():
        return int(numeric)
    return numeric


def _record_data(record: EvaluationRecord) -> dict[str, Any]:
    return {
        "input_format": record.input_format,
        "mlh": _number(record.mlh) if record.mlh is not None else None,
        "policy": [
            {
                "index": action.index,
                "logit": _number(action.logit),
                "move": action.move,
                "probability": _number(action.probability),
                "rank": action.rank,
            }
            for action in record.policy
        ],
        "position": {
            "chess960": record.position.chess960,
            "fen": record.position.fen,
            "moves": list(record.position.moves),
            "start_fen": record.position.start_fen,
            "variant": record.position.variant,
        },
        "provenance": {
            "model_type": record.provenance.model_type,
            "network": record.provenance.network,
            "network_checksum": record.provenance.network_checksum,
            "source": record.provenance.source,
        },
        "schema_version": record.schema_version,
        "value": (
            {
                "origin": record.value.origin.value,
                "perspective": record.value.perspective.value,
                "value": _number(record.value.value),
            }
            if record.value is not None
            else None
        ),
        "wdl": (
            {
                "draw": _number(record.wdl.draw),
                "loss": _number(record.wdl.loss),
                "perspective": record.wdl.perspective.value,
                "win": _number(record.wdl.win),
            }
            if record.wdl is not None
            else None
        ),
    }


def _decode_record(value: Any) -> EvaluationRecord:
    _require_fields(
        value,
        {"input_format", "mlh", "policy", "position", "provenance", "schema_version", "value", "wdl"},
        "EvaluationRecord",
    )
    if isinstance(value["schema_version"], bool) or value["schema_version"] != 1:
        raise EvaluationRecordFormatError(f"Unsupported EvaluationRecord schema version {value['schema_version']!r}.")
    position_data = value["position"]
    _require_fields(position_data, {"chess960", "fen", "moves", "start_fen", "variant"}, "PositionIdentity")
    provenance_data = value["provenance"]
    _require_fields(
        provenance_data,
        {"model_type", "network", "network_checksum", "source"},
        "EvaluationProvenance",
    )
    try:
        position = PositionIdentity(
            fen=_string(position_data["fen"], "PositionIdentity.fen"),
            start_fen=_string(position_data["start_fen"], "PositionIdentity.start_fen"),
            moves=_string_tuple(position_data["moves"], "PositionIdentity.moves"),
            variant=_string(position_data["variant"], "PositionIdentity.variant"),
            chess960=_boolean(position_data["chess960"], "PositionIdentity.chess960"),
        )
        provenance = EvaluationProvenance(
            source=_string(provenance_data["source"], "EvaluationProvenance.source"),
            model_type=_string(provenance_data["model_type"], "EvaluationProvenance.model_type"),
            network=_optional_string(provenance_data["network"], "EvaluationProvenance.network"),
            network_checksum=_optional_string(
                provenance_data["network_checksum"], "EvaluationProvenance.network_checksum"
            ),
        )
        policy_data = value["policy"]
        if not isinstance(policy_data, list):
            raise EvaluationRecordFormatError("EvaluationRecord.policy must be an array.")
        policy = tuple(_decode_action(action) for action in policy_data)
        wdl = _decode_wdl(value["wdl"])
        scalar = _decode_scalar(value["value"])
        mlh = None if value["mlh"] is None else _float(value["mlh"], "EvaluationRecord.mlh")
        return EvaluationRecord(
            position=position,
            provenance=provenance,
            input_format=_string(value["input_format"], "EvaluationRecord.input_format"),
            policy=policy,
            wdl=wdl,
            value=scalar,
            mlh=mlh,
        )
    except EvaluationRecordFormatError:
        raise
    except (TypeError, ValueError) as error:
        raise EvaluationRecordFormatError(f"Invalid EvaluationRecord: {error}") from error


def _decode_action(value: Any) -> ActionEvaluationRecord:
    _require_fields(value, {"index", "logit", "move", "probability", "rank"}, "ActionEvaluationRecord")
    return ActionEvaluationRecord(
        move=_string(value["move"], "ActionEvaluationRecord.move"),
        index=_integer(value["index"], "ActionEvaluationRecord.index"),
        logit=_float(value["logit"], "ActionEvaluationRecord.logit"),
        probability=_float(value["probability"], "ActionEvaluationRecord.probability"),
        rank=_integer(value["rank"], "ActionEvaluationRecord.rank"),
    )


def _decode_wdl(value: Any) -> WdlEvaluationRecord | None:
    if value is None:
        return None
    _require_fields(value, {"draw", "loss", "perspective", "win"}, "WdlEvaluationRecord")
    try:
        perspective = ChessPlayer(_string(value["perspective"], "WdlEvaluationRecord.perspective"))
    except ValueError as error:
        raise EvaluationRecordFormatError("Invalid WdlEvaluationRecord.perspective.") from error
    return WdlEvaluationRecord(
        win=_float(value["win"], "WdlEvaluationRecord.win"),
        draw=_float(value["draw"], "WdlEvaluationRecord.draw"),
        loss=_float(value["loss"], "WdlEvaluationRecord.loss"),
        perspective=perspective,
    )


def _decode_scalar(value: Any) -> ScalarEvaluationRecord | None:
    if value is None:
        return None
    _require_fields(value, {"origin", "perspective", "value"}, "ScalarEvaluationRecord")
    try:
        origin = ValueOrigin(_string(value["origin"], "ScalarEvaluationRecord.origin"))
        perspective = ChessPlayer(_string(value["perspective"], "ScalarEvaluationRecord.perspective"))
    except ValueError as error:
        raise EvaluationRecordFormatError("Invalid scalar origin or perspective.") from error
    return ScalarEvaluationRecord(
        value=_float(value["value"], "ScalarEvaluationRecord.value"),
        origin=origin,
        perspective=perspective,
    )


def _require_fields(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise EvaluationRecordFormatError(f"{label} must be an object.")
    actual = set(value)
    if actual != expected:
        raise EvaluationRecordFormatError(
            f"Invalid {label} fields: missing={sorted(expected - actual)!r}, unexpected={sorted(actual - expected)!r}."
        )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise EvaluationRecordFormatError(f"Duplicate JSON field {key!r}.")
        value[key] = item
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise EvaluationRecordFormatError(f"{label} must be a string.")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    return None if value is None else _string(value, label)


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EvaluationRecordFormatError(f"{label} must be an array of strings.")
    return tuple(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise EvaluationRecordFormatError(f"{label} must be a boolean.")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationRecordFormatError(f"{label} must be an integer.")
    return value


def _float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationRecordFormatError(f"{label} must be numeric.")
    try:
        result = float(value)
    except OverflowError as error:
        raise EvaluationRecordFormatError(f"{label} must be finite.") from error
    if not math.isfinite(result):
        raise EvaluationRecordFormatError(f"{label} must be finite.")
    return value if isinstance(value, int) else result


__all__ = [
    "EvaluationRecordFormatError",
    "deserialize_evaluation_record",
    "evaluation_record_digest",
    "serialize_evaluation_record",
]
