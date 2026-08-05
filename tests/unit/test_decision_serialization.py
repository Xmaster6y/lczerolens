"""Canonical persistence contracts for complete decision analyses."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import Any

import chess
import pytest

from lczerolens import DecisionAnalysis, DecisionAnalysisFormatError
from lczerolens.counterfactuals import sibling_counterfactual
from lczerolens.decision import DecisionAction, compare_counterfactual, compare_decision
from lczerolens.facts import ChessSide, FactPerspective, MaterialAnalyzer, SideSubject
from lczerolens.moves import analyze_line
from lczerolens.provenance import EvaluationProvenance
from lczerolens._decision_serialization import (
    _canonical_evidence,
    _canonical_json,
    _coerce_annotation,
    _decode_counterfactuals,
    _decode_evidence,
    _decode_lines,
    _reject_duplicate_keys,
    _require_fields,
    deserialize_decision_analysis,
    serialize_decision_analysis,
)
from tests.unit.test_decision import fixture_evaluator, fixture_search


def _decision() -> DecisionAnalysis:
    board = chess.Board()
    evaluator = fixture_evaluator()
    evaluator.provenance = EvaluationProvenance("decision-fixture", "tests.FixtureNetwork")
    lines = {
        move: analyze_line(
            board,
            (move,),
            MaterialAnalyzer(FactPerspective.WHITE),
            MaterialAnalyzer(FactPerspective.BLACK),
        )
        for move in ("d2d4", "e2e4")
    }
    comparison = compare_counterfactual(sibling_counterfactual(board, factual="e2e4", alternative="d2d4"), evaluator)
    return compare_decision(
        evaluator.evaluate(board),
        fixture_search(),
        line_analyses=lines,
        counterfactuals=(comparison,),
    )


def test_canonical_round_trip_digest_and_file_persistence(tmp_path):
    decision = _decision()
    encoded = decision.to_bytes()
    path = tmp_path / "decision.json"

    decision.save(path)
    restored = DecisionAnalysis.load(path)

    assert restored == decision
    assert DecisionAnalysis.from_bytes(encoded) == decision
    assert deserialize_decision_analysis(serialize_decision_analysis(decision)) == decision
    assert decision.digest() == hashlib.sha256(encoded).hexdigest()
    assert restored.digest() == decision.digest()
    assert restored.counterfactuals == decision.counterfactuals


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (b'"format_version":1', b'"format_version":2', "format version"),
        (b'"$type":"LineAnalysis"', b'"$type":"FutureAnalysis"', "record type"),
        (b'"lines":', b'"unexpected":null,"lines":', "unexpected"),
    ],
)
def test_incompatible_or_unknown_records_are_rejected(old, new, message):
    malformed = _decision().to_bytes().replace(old, new, 1)

    with pytest.raises(DecisionAnalysisFormatError, match=message):
        DecisionAnalysis.from_bytes(malformed)


def test_noncanonical_and_invalid_inputs_fail_closed():
    canonical = _decision().to_bytes()

    with pytest.raises(DecisionAnalysisFormatError, match="not canonical"):
        DecisionAnalysis.from_bytes(canonical + b"\n")
    with pytest.raises(DecisionAnalysisFormatError, match="Invalid decision-analysis JSON"):
        DecisionAnalysis.from_bytes(b"{")
    with pytest.raises(TypeError, match="expects bytes"):
        deserialize_decision_analysis("not bytes")
    with pytest.raises(TypeError, match="expects a DecisionAnalysis"):
        serialize_decision_analysis(object())
    with pytest.raises(DecisionAnalysisFormatError, match="Unsupported decision value"):
        _canonical_evidence(object())


def test_envelope_and_collection_shapes_fail_closed():
    canonical = _decision().to_bytes()
    duplicate = canonical.replace(b"{", b'{"format":"duplicate",', 1)
    wrong_format = canonical.replace(b"lczerolens.decision-analysis", b"future.decision-analysis", 1)
    bool_envelope = json.loads(canonical)
    bool_envelope["format_version"] = True

    with pytest.raises(DecisionAnalysisFormatError, match="Duplicate JSON field"):
        DecisionAnalysis.from_bytes(duplicate)
    with pytest.raises(DecisionAnalysisFormatError, match="format 'future"):
        DecisionAnalysis.from_bytes(wrong_format)
    with pytest.raises(DecisionAnalysisFormatError, match="format version True"):
        DecisionAnalysis.from_bytes(_canonical_json(bool_envelope))
    with pytest.raises(DecisionAnalysisFormatError, match="lines must be an array"):
        _decode_lines(None)
    with pytest.raises(DecisionAnalysisFormatError, match="counterfactuals must be an array"):
        _decode_counterfactuals(None)


def test_line_and_counterfactual_entries_are_strict_and_canonical():
    decision = _decision()
    line = _canonical_evidence(decision.actions["e2e4"].line)
    entry = {"line": line, "move": "e2e4"}

    with pytest.raises(DecisionAnalysisFormatError, match="moves must be strings"):
        _decode_lines([{"line": line, "move": 1}])
    with pytest.raises(DecisionAnalysisFormatError, match="LineAnalysis"):
        _decode_lines([{"line": _canonical_evidence(SideSubject(ChessSide.WHITE)), "move": "e2e4"}])
    with pytest.raises(DecisionAnalysisFormatError, match="Duplicate decision line"):
        _decode_lines([entry, entry])
    with pytest.raises(DecisionAnalysisFormatError, match="canonical UCI order"):
        _decode_lines([entry, {"line": line, "move": "d2d4"}])
    with pytest.raises(DecisionAnalysisFormatError, match="CounterfactualPair"):
        _decode_counterfactuals([{"alternative_evaluation": {}, "factual_evaluation": {}, "pair": line}])

    malformed = json.loads(decision.to_bytes())
    malformed["counterfactuals"][0]["factual_evaluation"] = malformed["evaluation"]
    with pytest.raises(DecisionAnalysisFormatError, match="factual evaluation must match"):
        DecisionAnalysis.from_bytes(_canonical_json(malformed))


def test_typed_evidence_codec_rejects_unknown_or_invalid_values():
    class FutureEnum(Enum):
        VALUE = "value"

    @dataclass(frozen=True)
    class FutureRecord:
        value: int

    with pytest.raises(DecisionAnalysisFormatError, match="Unsupported decision enum"):
        _canonical_evidence(FutureEnum.VALUE)
    with pytest.raises(DecisionAnalysisFormatError, match="Unsupported decision record"):
        _canonical_evidence(FutureRecord(1))
    with pytest.raises(DecisionAnalysisFormatError, match="non-finite"):
        _canonical_evidence(math.inf)
    assert _canonical_evidence(-0.0) == 0
    assert _canonical_evidence(1.5) == 1.5

    malformed = (
        ({}, "exactly one supported type tag"),
        ({"$chess_move": "bad"}, "Invalid canonical chess move"),
        ({"$chess_piece": {"color": True, "piece_type": "pawn"}}, "Invalid canonical chess piece"),
        ({"$enum": "FutureEnum", "value": "value"}, "Unknown decision enum"),
        ({"$enum": "ChessSide", "value": "green"}, "Invalid ChessSide"),
        ({"$type": "FutureRecord"}, "Unknown decision record type"),
        ({"$type": "SideSubject"}, "Invalid SideSubject fields"),
        ({"$type": "SideSubject", "side": "white"}, "declared field type"),
        (
            {"$type": "RegionSubject", "name": "", "squares": [0]},
            "Invalid RegionSubject record",
        ),
    )
    for value, message in malformed:
        with pytest.raises(DecisionAnalysisFormatError, match=message):
            _decode_evidence(value)


def test_annotation_and_json_helpers_reject_ambiguous_values():
    assert _decode_evidence([1, 2]) == (1, 2)
    assert _coerce_annotation("anything", Any) == "anything"
    assert _coerce_annotation(1, float) == 1.0
    assert _coerce_annotation(None, type(None)) is None
    assert _coerce_annotation(ChessSide.WHITE, ChessSide) is ChessSide.WHITE
    assert _coerce_annotation((1, "two"), tuple[int, str]) == (1, "two")

    for value, annotation in (
        ("bad", int | None),
        ([1], tuple[int, ...]),
        ((1,), tuple[int, int]),
        (True, float),
        (True, int),
        (1, bool),
        (1, str),
        (1, type(None)),
        (1, DecisionAction),
    ):
        with pytest.raises(TypeError):
            _coerce_annotation(value, annotation)

    with pytest.raises(DecisionAnalysisFormatError, match="not canonical JSON"):
        _canonical_json(object())
    with pytest.raises(DecisionAnalysisFormatError, match="must be an object"):
        _require_fields(None, set(), "Fixture")
    with pytest.raises(DecisionAnalysisFormatError, match="missing=.*field"):
        _require_fields({}, {"field"}, "Fixture")
    with pytest.raises(DecisionAnalysisFormatError, match="Duplicate JSON field"):
        _reject_duplicate_keys([("field", 1), ("field", 2)])
