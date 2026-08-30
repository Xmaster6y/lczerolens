"""Runtime-to-evidence and canonical evaluation persistence contracts."""

from dataclasses import fields, is_dataclass, replace
import hashlib
import json

import chess
import chess.variant
import pytest
import torch
from tensordict import TensorDictBase
from torch import nn

from lczerolens import EvaluationRecord, EvaluationRecordFormatError, LczeroEvaluator, LczeroModel
from lczerolens._codec import encode_move
from lczerolens.evaluation import (
    ActionEvaluationRecord,
    EvaluationDerivation,
    ScalarEvaluationRecord,
    ValueOrigin,
    WdlEvaluationRecord,
)
from lczerolens.provenance import ChessPlayer, EvaluationProvenance, PositionIdentity
from lczerolens.serialization import deserialize_evaluation_record, serialize_evaluation_record


FIXTURE_DIGEST = "ae4dd00f10e6d2ac30a3418f0e962fff10b9a318317442c6d51791f7602d9f16"


class RecordNetwork(nn.Module):
    def forward(self, planes):
        batch = planes.shape[0]
        board = chess.Board()
        policy = torch.zeros((batch, 1858), device=planes.device)
        policy[:, encode_move(board, chess.Move.from_uci("e2e4"))] = 4
        wdl = torch.tensor([[0.6, 0.3, 0.1]], device=planes.device).repeat(batch, 1)
        mlh = torch.full((batch,), 24.0, device=planes.device)
        return policy, wdl, mlh


def record_evaluator(*, provenance=None):
    model = LczeroModel(RecordNetwork(), out_keys=["policy", "wdl", "mlh"])
    return LczeroEvaluator(model, provenance=provenance)


def test_record_freezes_runtime_values_position_history_and_provenance():
    provenance = EvaluationProvenance(
        source="fixture-evaluator",
        model_type="tests.RecordNetwork",
        network="fixture.pt",
        network_checksum=f"sha256:{'a' * 64}",
    )
    evaluator = record_evaluator(provenance=provenance)
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    evaluation = evaluator.evaluate(board)

    record = evaluation.record()
    original_bytes = record.to_bytes()
    evaluation.tensors["network", "policy_logits"].zero_()
    evaluation.tensors["evaluation", "policy"].fill_(0)

    assert record.position.fen == board.fen(en_passant="fen")
    assert record.position.moves == ("e2e4", "e7e5")
    assert record.position.board().fen(en_passant="fen") == board.fen(en_passant="fen")
    assert tuple(record.position.board().move_stack) == tuple(board.move_stack)
    assert record.provenance is provenance
    assert record.input_format == "classical_112"
    assert len(record.policy) == 29
    assert sum(action.probability for action in record.policy) == pytest.approx(1)
    assert record.wdl.perspective is ChessPlayer.WHITE
    assert (record.wdl.win, record.wdl.draw, record.wdl.loss) == pytest.approx((0.6, 0.3, 0.1))
    assert record.value.origin is ValueOrigin.DERIVED_FROM_WDL
    assert record.value.perspective is ChessPlayer.WHITE
    assert record.value.value == pytest.approx(0.5)
    assert record.mlh == pytest.approx(24)
    assert record.to_bytes() == original_bytes
    assert not _contains_runtime_state(record)


def test_terminal_record_has_an_explicit_empty_policy():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    record = record_evaluator().evaluate(board).record()

    assert record.policy == ()
    assert record.position.player is ChessPlayer.BLACK
    assert record.value.perspective is ChessPlayer.BLACK


def test_canonical_round_trip_digest_and_file_persistence(tmp_path):
    record = record_evaluator().evaluate(chess.Board()).record()
    encoded = record.to_bytes()
    path = tmp_path / "evaluation.json"

    record.save(path)
    restored = EvaluationRecord.load(path)

    assert restored == record
    assert EvaluationRecord.from_bytes(encoded) == record
    assert deserialize_evaluation_record(serialize_evaluation_record(record)) == record
    assert record.digest() == hashlib.sha256(encoded).hexdigest()
    assert restored.digest() == record.digest()


def test_canonical_evaluation_digest_is_stable():
    provenance = EvaluationProvenance(
        "digest-fixture",
        "tests.RecordNetwork",
        "fixture.pt",
        f"sha256:{'a' * 64}",
    )
    record = record_evaluator(provenance=provenance).evaluate(chess.Board()).record()

    assert record.digest() == FIXTURE_DIGEST


def test_fixture_backed_torch_loading_records_network_checksum(tmp_path):
    path = tmp_path / "fixture.pt"
    torch.save(LczeroModel(RecordNetwork(), out_keys=["policy", "wdl", "mlh"]), path)

    evaluator = LczeroEvaluator.from_path(str(path), weights_only=False)
    record = evaluator.evaluate(chess.Board()).record()

    assert record.provenance.network == "fixture.pt"
    assert record.provenance.network_checksum == f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    assert record.provenance.model_type.endswith("RecordNetwork")


@pytest.mark.parametrize(
    "board",
    [
        chess.Board.from_chess960_pos(0),
        chess.variant.AtomicBoard(),
    ],
)
def test_position_identity_preserves_variants(board):
    identity = PositionIdentity.from_board(board)
    restored = identity.board()

    assert type(restored) is type(board)
    assert restored.chess960 is board.chess960
    assert restored.fen() == board.fen()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (b'"format":"lczerolens.evaluation"', b'"format":"future"', "format"),
        (b'"format_version":1', b'"format_version":2', "format version"),
        (b'"schema_version":1', b'"schema_version":3', "schema version"),
        (b'"input_format":"classical_112"', b'"input_format":1', "input_format"),
        (b'"perspective":"white"', b'"perspective":"green"', "perspective"),
    ],
)
def test_incompatible_or_malformed_evaluation_bytes_are_rejected(old, new, message):
    encoded = record_evaluator().evaluate(chess.Board()).record().to_bytes()

    with pytest.raises(EvaluationRecordFormatError, match=message):
        EvaluationRecord.from_bytes(encoded.replace(old, new, 1))


def test_duplicate_noncanonical_and_invalid_utf8_bytes_are_rejected():
    encoded = record_evaluator().evaluate(chess.Board()).record().to_bytes()
    duplicate = encoded.replace(b'{"format":', b'{"format":"lczerolens.evaluation","format":', 1)

    with pytest.raises(EvaluationRecordFormatError, match="Duplicate"):
        EvaluationRecord.from_bytes(duplicate)
    with pytest.raises(EvaluationRecordFormatError, match="not canonical"):
        EvaluationRecord.from_bytes(encoded + b"\n")
    with pytest.raises(EvaluationRecordFormatError, match="Invalid evaluation-record JSON"):
        EvaluationRecord.from_bytes(b"\xff")


def test_serialization_entry_points_require_documented_types():
    with pytest.raises(TypeError, match="EvaluationRecord"):
        serialize_evaluation_record("record")
    with pytest.raises(TypeError, match="bytes"):
        deserialize_evaluation_record("record")


def test_derived_evaluation_metadata_round_trips_and_version_one_remains_readable():
    evaluation = record_evaluator().evaluate(chess.Board())
    logits = evaluation.tensors["network", "policy_logits"].clone()
    derived = evaluation.derive(policy_logits=logits, value=-0.25).record()

    restored = EvaluationRecord.from_bytes(derived.to_bytes())
    version_one = evaluation.record()
    version_one_bytes = version_one.to_bytes()

    assert derived.schema_version == 2
    assert restored.derivation == EvaluationDerivation(policy_logits_replaced=True, value_replaced=True)
    assert restored.value.origin is ValueOrigin.DERIVED
    assert version_one.schema_version == 1
    assert EvaluationRecord.from_bytes(version_one_bytes) == version_one
    assert b'"derivation"' not in version_one.to_bytes()


def test_record_validation_fails_closed_for_position_policy_and_provenance():
    board = chess.Board()
    position = PositionIdentity.from_board(board)
    provenance = EvaluationProvenance("fixture", "tests.Model")
    action = ActionEvaluationRecord("e2e4", 0, 1.0, 1.0, 1)

    with pytest.raises(ValueError, match="exactly match"):
        EvaluationRecord(position, provenance, "classical_112", (action,))
    with pytest.raises(ValueError, match="input format"):
        EvaluationRecord(position, provenance, "future", ())
    with pytest.raises(ValueError, match="source"):
        EvaluationProvenance("", "tests.Model")
    with pytest.raises(ValueError, match="model_type"):
        EvaluationProvenance("fixture", "")
    with pytest.raises(ValueError, match="checksum"):
        EvaluationProvenance("fixture", "tests.Model", network_checksum="sha256:bad")
    with pytest.raises(ValueError, match="reconstruct"):
        replace(position, fen=chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1").fen())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"move": "bad"}, "valid UCI"),
        ({"index": True}, "indices"),
        ({"logit": float("inf")}, "logits"),
        ({"probability": -0.1}, "probabilities"),
        ({"rank": 0}, "ranks"),
    ],
)
def test_action_record_validation(kwargs, message):
    values = {"move": "e2e4", "index": 0, "logit": 1.0, "probability": 1.0, "rank": 1}
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ActionEvaluationRecord(**values)


def test_scalar_and_wdl_record_validation():
    with pytest.raises(ValueError, match="scalar values"):
        ScalarEvaluationRecord(2.0, ValueOrigin.NATIVE, ChessPlayer.WHITE)
    with pytest.raises(ValueError, match="origins"):
        ScalarEvaluationRecord(0.0, "native", ChessPlayer.WHITE)
    with pytest.raises(ValueError, match="perspectives"):
        ScalarEvaluationRecord(0.0, ValueOrigin.NATIVE, "white")
    with pytest.raises(ValueError, match="finite probabilities"):
        WdlEvaluationRecord(-0.1, 0.5, 0.6, ChessPlayer.WHITE)
    with pytest.raises(ValueError, match="sum to one"):
        WdlEvaluationRecord(0.5, 0.5, 0.5, ChessPlayer.WHITE)
    with pytest.raises(ValueError, match="perspectives"):
        WdlEvaluationRecord(0.5, 0.3, 0.2, "white")


def test_evaluation_record_validation_rejects_incoherent_evidence():
    base = record_evaluator().evaluate(chess.Board()).record()
    other_player = ChessPlayer.BLACK

    for changes, message in (
        ({"position": "not a position"}, "PositionIdentity"),
        ({"provenance": "not provenance"}, "EvaluationProvenance"),
        ({"input_format": ""}, "input format"),
        ({"policy": ("not an action",)}, "policy entries"),
        ({"wdl": "not wdl"}, "WDL"),
        ({"value": "not value"}, "record value"),
        ({"derivation": "not derivation"}, "record derivation"),
        ({"policy": (base.policy[0], base.policy[0], *base.policy[1:])}, "unique"),
        ({"policy": tuple(reversed(base.policy))}, "canonical UCI order"),
        ({"policy": (replace(base.policy[0], index=1), *base.policy[1:])}, "indices must match"),
        ({"policy": (replace(base.policy[0], rank=1), *base.policy[1:])}, "ranks must agree"),
        (
            {"policy": tuple(replace(action, probability=0.0) for action in base.policy)},
            "sum to one",
        ),
        ({"wdl": replace(base.wdl, perspective=other_player)}, "WDL perspective"),
        ({"value": replace(base.value, perspective=other_player)}, "scalar perspective"),
        ({"mlh": float("inf")}, "MLH"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(base, **changes)


def test_evaluation_derivation_requires_boolean_replacements():
    with pytest.raises(TypeError, match="booleans"):
        EvaluationDerivation(policy_logits_replaced=1)
    with pytest.raises(ValueError, match="replace"):
        EvaluationDerivation()


def test_position_identity_rejects_invalid_inputs_and_history():
    start = chess.Board().fen()

    with pytest.raises(TypeError, match="python-chess"):
        PositionIdentity.from_board("not a board")
    with pytest.raises(ValueError, match="variant"):
        PositionIdentity(start, start, (), variant="")
    with pytest.raises(ValueError, match="fen must"):
        PositionIdentity(1, start, ())
    with pytest.raises(ValueError, match="start_fen"):
        PositionIdentity(start, 1, ())
    with pytest.raises(ValueError, match="moves must be a tuple"):
        PositionIdentity(start, start, [])
    with pytest.raises(ValueError, match="chess960"):
        PositionIdentity(start, start, (), chess960=1)
    with pytest.raises(ValueError, match="legal sequence"):
        PositionIdentity(start, start, ("e2e5",))
    with pytest.raises(ValueError, match="legal sequence"):
        PositionIdentity(start, start, (), variant="future")


def test_evaluation_provenance_rejects_non_string_fields():
    for kwargs, message in (
        ({"source": 1, "model_type": "fixture"}, "source"),
        ({"source": "fixture", "model_type": 1}, "model_type"),
        ({"source": "fixture", "model_type": "fixture", "network": 1}, "network"),
        ({"source": "fixture", "model_type": "fixture", "network_checksum": 1}, "network_checksum"),
    ):
        with pytest.raises(ValueError, match=message):
            EvaluationProvenance(**kwargs)


def test_malformed_nested_field_types_are_rejected():
    record = record_evaluator().evaluate(chess.Board()).record()
    encoded = record.to_bytes()

    malformed_cases = (
        (encoded.replace(b'"record":{', b'"record":null,"ignored":{', 1), "fields"),
        (encoded.replace(b'"policy":[', b'"policy":null,"ignored":[', 1), "fields"),
        (encoded.replace(b'"moves":[]', b'"moves":[1]', 1), "moves"),
        (encoded.replace(b'"chess960":false', b'"chess960":1', 1), "chess960"),
        (encoded.replace(b'"index":', b'"index":true,"ignored":', 1), "fields"),
        (encoded.replace(b'"logit":0', b'"logit":"zero"', 1), "numeric"),
    )
    for malformed, message in malformed_cases:
        with pytest.raises(EvaluationRecordFormatError, match=message):
            EvaluationRecord.from_bytes(malformed)


def test_canonical_decoder_rejects_structurally_typed_but_invalid_values():
    encoded = record_evaluator().evaluate(chess.Board()).record().to_bytes()

    def malformed(mutator):
        value = json.loads(encoded)
        mutator(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()

    cases = (
        (lambda value: value.update(record=None), "EvaluationRecord must be an object"),
        (lambda value: value["record"].update(policy=None), "policy must be an array"),
        (
            lambda value: value["record"]["provenance"].update(network_checksum="bad"),
            "Invalid EvaluationRecord",
        ),
        (lambda value: value["record"]["wdl"].update(perspective="green"), "WdlEvaluationRecord"),
        (lambda value: value["record"]["policy"][0].update(index=True), "index must be an integer"),
    )
    for mutator, message in cases:
        with pytest.raises(EvaluationRecordFormatError, match=message):
            EvaluationRecord.from_bytes(malformed(mutator))

    nonfinite = encoded.replace(b'"logit":0', b'"logit":1e999', 1)
    with pytest.raises(EvaluationRecordFormatError, match="finite"):
        EvaluationRecord.from_bytes(nonfinite)


def test_absent_optional_heads_round_trip_and_utf8_is_strict():
    record = record_evaluator().evaluate(chess.Board()).record()
    minimal = replace(record, wdl=None, value=None, mlh=None)

    assert EvaluationRecord.from_bytes(minimal.to_bytes()) == minimal

    malformed_text = replace(record, provenance=replace(record.provenance, source="\ud800"))
    with pytest.raises(EvaluationRecordFormatError, match="valid UTF-8"):
        malformed_text.to_bytes()

    object.__setattr__(minimal, "mlh", float("inf"))
    with pytest.raises(EvaluationRecordFormatError, match="non-finite"):
        minimal.to_bytes()

    object.__setattr__(minimal, "mlh", True)
    with pytest.raises(EvaluationRecordFormatError, match="numeric evidence"):
        minimal.to_bytes()


def test_large_integer_evidence_round_trips_losslessly_and_overflow_is_wrapped():
    record = record_evaluator().evaluate(chess.Board()).record()
    selected = next(action for action in record.policy if action.rank == 1)
    exact = 2**60 + 1
    policy = tuple(replace(action, logit=exact) if action is selected else action for action in record.policy)
    large = replace(record, policy=policy)

    restored = EvaluationRecord.from_bytes(large.to_bytes())
    assert next(action for action in restored.policy if action.move == selected.move).logit == exact

    malformed = record.to_bytes().replace(b'"logit":0', b'"logit":' + b"9" * 400, 1)
    with pytest.raises(EvaluationRecordFormatError, match="finite"):
        EvaluationRecord.from_bytes(malformed)


def _contains_runtime_state(value):
    if isinstance(value, torch.Tensor | TensorDictBase | nn.Module):
        return True
    if is_dataclass(value) and not isinstance(value, type):
        return any(_contains_runtime_state(getattr(value, item.name)) for item in fields(value))
    if isinstance(value, tuple):
        return any(_contains_runtime_state(item) for item in value)
    return False
