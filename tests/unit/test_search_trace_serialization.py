"""Canonical persistence contracts for search traces."""

from dataclasses import replace
import math

import pytest

import lczerolens.search_trace as search_trace_module
from lczerolens.search_trace import (
    BackupUpdate,
    ChessPlayer,
    EdgeStatistics,
    EvaluatorCall,
    LeafRecord,
    NodeExpansion,
    PathStep,
    PositionEvaluation,
    PrincipalVariation,
    RootAction,
    RootSelection,
    RootSnapshot,
    SearchBudget,
    SearchBudgetUnit,
    SearchCapability,
    SearchParameter,
    SearchProvenance,
    SearchTrace,
    SearchTraceFormatError,
    SimulationEvent,
    ValuePerspective,
    Wdl,
    deserialize_search_trace,
    search_trace_digest,
    serialize_search_trace,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FIXTURE_DIGEST = "144a0c0cca94ecaf696781b38da7f8aefed881b2c9c8335dcef7c854d0b7b8c0"


def _full_trace() -> SearchTrace:
    e2e4_before = EdgeStatistics(
        "e2e4",
        ValuePerspective.ROOT_PLAYER,
        prior=0.6,
        visits=0,
        total_value=0.0,
        mean_value=0.0,
        exploration=0.2,
    )
    e2e4_after = replace(e2e4_before, visits=1, total_value=0.25, mean_value=0.25, exploration=0.1)
    d2d4 = EdgeStatistics(
        "d2d4",
        ValuePerspective.ROOT_PLAYER,
        prior=0.4,
        visits=0,
        total_value=0.0,
        mean_value=0.0,
        exploration=0.15,
    )
    leaf_evaluation = PositionEvaluation(
        ValuePerspective.ROOT_PLAYER,
        value=0.25,
        wdl=Wdl(0.5, 0.25, 0.25, ValuePerspective.ROOT_PLAYER),
    )
    evaluator = EvaluatorCall(
        dtype="float32",
        source_device="mps:0",
        search_device="cpu",
        legal_policy_logits=(("e7e5", 2.5), ("c7c5", -0.5)),
    )
    event = SimulationEvent(
        event_id="simulation-0",
        simulation=0,
        path=(PathStep("root", "e2e4", "node-e2e4"),),
        leaf=LeafRecord("node-e2e4", leaf_evaluation, terminal=False, evaluator=evaluator),
        backups=(BackupUpdate("root", 0.25, e2e4_before, e2e4_after),),
        expansion=NodeExpansion(
            "node-e2e4",
            (EdgeStatistics("e7e5", ValuePerspective.SIDE_TO_MOVE, prior=1.0),),
        ),
        root_before=(e2e4_before, d2d4),
        root_after=(e2e4_after, d2d4),
    )
    return SearchTrace(
        root_fen=START_FEN,
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.REPLAYABLE,
        provenance=SearchProvenance(
            source="reference-search",
            engine="lczerolens",
            engine_version="0.4.0",
            network="fixture.pt",
            network_checksum="sha256:0123456789abcdef",
            parameters=(
                SearchParameter("enabled", True),
                SearchParameter("optional", None),
                SearchParameter("seed", 7),
                SearchParameter("temperature", 1.5),
                SearchParameter("variant", "standard"),
            ),
        ),
        snapshots=(
            RootSnapshot(
                sequence=0,
                selection=RootSelection("e2e4", "maximum N", "UCI order", temperature=0.0),
                evaluation=leaf_evaluation,
                budget=SearchBudget(SearchBudgetUnit.SIMULATIONS, requested=1, observed=1),
                actions=(
                    RootAction(
                        e2e4_after,
                        evaluation=leaf_evaluation,
                        leaf_evaluation=leaf_evaluation,
                        principal_variation=PrincipalVariation(("e2e4", "e7e5")),
                    ),
                    RootAction(d2d4),
                ),
            ),
        ),
        events=(event,),
        root_expansion=NodeExpansion("root", (e2e4_before, d2d4)),
        root_evaluator=evaluator,
        root_start_fen=START_FEN,
        root_move_history=(),
    )


@pytest.mark.parametrize("capability", list(SearchCapability))
def test_every_capability_round_trips_without_upgrading(capability):
    trace = replace(_full_trace(), capability=capability)

    loaded = deserialize_search_trace(serialize_search_trace(trace))

    assert loaded == trace
    assert loaded.capability is capability


def test_absent_and_empty_collections_remain_distinct():
    absent = SearchTrace(
        root_fen=START_FEN,
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.ROOT_RESULT,
        provenance=SearchProvenance("fixture"),
        snapshots=(RootSnapshot(0, evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0)),),
    )
    empty = replace(absent, snapshots=(replace(absent.snapshots[0], actions=()),), events=())

    assert serialize_search_trace(absent) != serialize_search_trace(empty)
    assert deserialize_search_trace(serialize_search_trace(absent)).snapshots[0].actions is None
    assert deserialize_search_trace(serialize_search_trace(empty)).snapshots[0].actions == ()
    assert deserialize_search_trace(serialize_search_trace(absent)).events is None
    assert deserialize_search_trace(serialize_search_trace(empty)).events == ()


def test_equivalent_negative_zero_values_have_identical_bytes_and_digest():
    positive = SearchTrace(
        root_fen=START_FEN,
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.ROOT_RESULT,
        provenance=SearchProvenance("fixture"),
        snapshots=(RootSnapshot(0, evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0)),),
    )
    negative = replace(
        positive,
        snapshots=(RootSnapshot(0, evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=-0.0)),),
    )

    assert positive == negative
    assert serialize_search_trace(positive) == serialize_search_trace(negative)
    assert search_trace_digest(positive) == search_trace_digest(negative)


def test_equivalent_integral_float_values_have_identical_bytes():
    integer = replace(
        _full_trace(),
        provenance=replace(
            _full_trace().provenance,
            parameters=(SearchParameter("temperature", 1),),
        ),
    )
    floating = replace(
        integer,
        provenance=replace(
            integer.provenance,
            parameters=(SearchParameter("temperature", 1.0),),
        ),
    )

    assert integer == floating
    assert serialize_search_trace(integer) == serialize_search_trace(floating)


def test_canonical_fixture_and_expected_digest_are_stable():
    fixture = serialize_search_trace(_full_trace())

    assert search_trace_digest(_full_trace()) == FIXTURE_DIGEST
    assert deserialize_search_trace(fixture) == _full_trace()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (b'"format_version":1', b'"format_version":2', "format version"),
        (b'"schema_version":1', b'"schema_version":2', "schema version"),
        (b'"$type":"SearchTrace"', b'"$type":"FutureTrace"', "record type"),
        (b'"root_fen":', b'"unexpected":null,"root_fen":', "unexpected"),
    ],
)
def test_incompatible_or_unknown_records_are_rejected(old, new, message):
    malformed = serialize_search_trace(_full_trace()).replace(old, new, 1)

    with pytest.raises(SearchTraceFormatError, match=message):
        deserialize_search_trace(malformed)


@pytest.mark.parametrize("value", [math.inf, object()])
def test_unsupported_canonical_values_are_rejected(value):
    with pytest.raises(SearchTraceFormatError, match="non-finite|Unsupported"):
        search_trace_module._canonical_record(value)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"$enum": "ChessPlayer", "value": "white", "extra": None}, "exactly"),
        ({"$enum": 1, "value": "white"}, "enum names"),
        ({"$enum": "FutureEnum", "value": "white"}, "Unknown search-trace enum"),
        ({"$enum": "ChessPlayer", "value": "green"}, "Invalid ChessPlayer"),
        ({"value": "white"}, "require a '\\$type'"),
        ({"$type": 1}, "type names"),
        ({"$type": "SearchParameter", "name": "fixture"}, "missing"),
        ({"$type": "SearchParameter", "name": "", "value": 1}, "Invalid SearchParameter"),
    ],
)
def test_malformed_nested_records_are_rejected(record, message):
    with pytest.raises(SearchTraceFormatError, match=message):
        search_trace_module._decode_record(record)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"\xff", "Invalid search-trace JSON"),
        (b"{", "Invalid search-trace JSON"),
        (b"null", "envelopes require"),
        (b"{}", "envelopes require"),
        (
            b'{"format":"another-format","format_version":1,"trace":null}',
            "Unsupported search-trace format",
        ),
        (
            b'{"format":"lczerolens.search-trace","format_version":true,"trace":null}',
            "format version",
        ),
        (
            b'{"format":"lczerolens.search-trace","format_version":1,"trace":null}',
            "must contain a SearchTrace",
        ),
    ],
)
def test_malformed_envelopes_are_rejected(data, message):
    with pytest.raises(SearchTraceFormatError, match=message):
        deserialize_search_trace(data)


def test_boolean_schema_version_is_rejected():
    malformed = serialize_search_trace(_full_trace()).replace(b'"schema_version":1', b'"schema_version":true', 1)

    with pytest.raises(SearchTraceFormatError, match="SearchTrace.schema_version"):
        deserialize_search_trace(malformed)


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        (b'"terminal":false', b'"terminal":"false"', "LeafRecord.terminal"),
        (
            b'{"$enum":"SearchBudgetUnit","value":"simulations"}',
            b'"simulations"',
            "SearchBudget.unit",
        ),
        (b'"simulation":0', b'"simulation":true', "SimulationEvent.simulation"),
        (
            b'{"$type":"SearchParameter","name":"enabled","value":true}',
            b"null",
            "SearchProvenance.parameters",
        ),
    ],
)
def test_declared_field_types_cannot_be_bypassed(old, new, field):
    malformed = serialize_search_trace(_full_trace()).replace(old, new, 1)

    with pytest.raises(SearchTraceFormatError, match=field):
        deserialize_search_trace(malformed)


def test_integral_json_numbers_are_restored_to_declared_float_fields():
    loaded = deserialize_search_trace(serialize_search_trace(_full_trace()))

    assert type(loaded.events[0].root_before[0].total_value) is float
    assert type(loaded.snapshots[0].budget.requested) is int


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        (None, tuple[str, ...]),
        (("only-one",), tuple[str, int]),
        ("not-numeric", float | None),
        (1, str),
    ],
)
def test_annotation_validation_fails_closed_for_every_supported_shape(value, annotation):
    with pytest.raises(TypeError):
        search_trace_module._coerce_annotation(value, annotation)


def test_lone_surrogates_are_rejected_at_the_utf8_boundary():
    trace = _full_trace()
    malformed_trace = replace(trace, provenance=replace(trace.provenance, source="\ud800"))
    with pytest.raises(SearchTraceFormatError, match="valid UTF-8"):
        serialize_search_trace(malformed_trace)

    malformed_bytes = serialize_search_trace(trace).replace(b'"source":"reference-search"', b'"source":"\\ud800"')
    with pytest.raises(SearchTraceFormatError, match="valid UTF-8"):
        deserialize_search_trace(malformed_bytes)


def test_duplicate_and_noncanonical_json_are_rejected():
    duplicate = serialize_search_trace(_full_trace()).replace(
        b'{"format":', b'{"format":"lczerolens.search-trace","format":', 1
    )
    with pytest.raises(SearchTraceFormatError, match="Duplicate JSON field"):
        deserialize_search_trace(duplicate)
    with pytest.raises(SearchTraceFormatError, match="not canonical"):
        deserialize_search_trace(serialize_search_trace(_full_trace()) + b"\n")


def test_serialization_entry_points_require_their_documented_types():
    with pytest.raises(TypeError, match="SearchTrace"):
        serialize_search_trace("not a trace")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes"):
        deserialize_search_trace("not bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported value type"):
        SearchParameter("nested", ())  # type: ignore[arg-type]
