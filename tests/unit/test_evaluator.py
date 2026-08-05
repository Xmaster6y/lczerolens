"""Contract tests for the TensorDict-centered Lczero evaluator."""

import chess
import pytest
import torch
from tensordict import TensorDict
from torch import nn

from lczerolens import LczeroModel
from lczerolens._codec import encode_move
from lczerolens.evaluation import Evaluation, EvaluationBatch, ValueOrigin
from lczerolens.evaluator import LczeroEvaluator


class FixtureNetwork(nn.Module):
    def __init__(self, *, wdl: bool = False):
        super().__init__()
        self.wdl = wdl

    def forward(self, planes):
        batch = planes.shape[0]
        policy = torch.zeros((batch, 1858), device=planes.device)
        board = chess.Board()
        policy[:, encode_move(board, chess.Move.from_uci("e2e4"))] = 4
        policy[:, encode_move(board, chess.Move.from_uci("d2d4"))] = 2
        if self.wdl:
            return policy, torch.tensor([[0.6, 0.3, 0.1]], device=planes.device).repeat(batch, 1)
        return policy, torch.full((batch,), 0.25, device=planes.device)


class ShiftPolicyModel(LczeroModel):
    """Fixture proving that evaluator execution preserves model behavior."""

    def _call_module(self, tensors, **kwargs):
        policy, value = super()._call_module(tensors, **kwargs)
        return policy + 1, value


class PolicyOnlyNetwork(nn.Module):
    def forward(self, planes):
        return torch.zeros((planes.shape[0], 1858), device=planes.device)


def fixture_evaluator(*, wdl=False):
    heads = ["policy", "wdl" if wdl else "value"]
    return LczeroEvaluator(LczeroModel(FixtureNetwork(wdl=wdl), out_keys=heads))


def test_evaluator_preserves_lczero_model_subclass_execution():
    model = ShiftPolicyModel(FixtureNetwork(), out_keys=["policy", "value"])
    evaluation = LczeroEvaluator(model).evaluate(chess.Board())

    assert evaluation.tensors["network", "policy_logits"].min().item() == pytest.approx(1)
    assert evaluation.policy["e2e4"].logit == pytest.approx(5)


def test_evaluate_one_position_has_natural_legal_policy_and_native_value():
    evaluation = fixture_evaluator().evaluate(chess.Board())

    assert evaluation.policy.best_move == chess.Move.from_uci("e2e4")
    assert evaluation.policy["e2e4"].probability > evaluation.policy["d2d4"].probability
    assert evaluation.policy["e2e4"].rank == 1
    assert evaluation.policy.top(2)[0].move == chess.Move.from_uci("e2e4")
    assert evaluation.value is not None
    assert evaluation.value.value == pytest.approx(0.25)
    assert evaluation.value.origin is ValueOrigin.NATIVE
    assert evaluation.value.perspective == chess.WHITE
    assert evaluation.wdl is None
    assert evaluation.mlh is None
    assert len(evaluation.policy.actions) == 20
    assert evaluation.position is not evaluation.position

    with pytest.raises(KeyError, match="not a legal evaluated move"):
        evaluation.policy["e2e5"]
    with pytest.raises(ValueError, match="non-negative integer"):
        evaluation.policy.top(True)


def test_wdl_value_is_derived_without_overwriting_the_network_head():
    evaluation = fixture_evaluator(wdl=True).evaluate(chess.Board())

    assert evaluation.wdl is not None
    assert evaluation.wdl.win == pytest.approx(0.6)
    assert evaluation.wdl.perspective == chess.WHITE
    assert evaluation.value is not None
    assert evaluation.value.value == pytest.approx(0.5)
    assert evaluation.value.origin is ValueOrigin.DERIVED_FROM_WDL
    assert ("network", "value") not in evaluation.tensors.keys(include_nested=True, leaves_only=True)


def test_prepare_and_finish_preserve_batch_device_and_instrumentation_keys():
    evaluator = fixture_evaluator()
    boards = [chess.Board(), chess.Board()]
    tensors = evaluator.prepare(boards)

    assert tensors.batch_size == torch.Size([2])
    assert tensors["input", "planes"].shape == (2, 112, 8, 8)
    assert tensors["input", "legal_mask"].shape == (2, 1858)

    tensors = evaluator.model(tensors)
    tensors["attr", "input", "planes"] = torch.ones_like(tensors["input", "planes"])
    evaluations = evaluator.finish(boards, tensors)

    assert len(evaluations) == 2
    assert isinstance(evaluations[1:], EvaluationBatch)
    assert len(evaluations[1:]) == 1
    assert evaluations.tensors is tensors
    assert ("attr", "input", "planes") in tensors.keys(include_nested=True, leaves_only=True)
    assert torch.allclose(tensors["evaluation", "policy"].sum(-1), torch.ones(2))
    assert len(list(evaluations)) == 2


def test_terminal_position_retains_raw_heads_without_a_legal_policy():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    evaluation = fixture_evaluator().evaluate(board)

    assert not evaluation.policy.is_defined
    assert evaluation.policy.best_move is None
    assert not evaluation.tensors["evaluation", "policy"].any()
    assert evaluation.value is not None


def test_finish_rejects_malformed_or_position_incompatible_tensors():
    evaluator = fixture_evaluator()
    board = chess.Board()
    tensors = evaluator.prepare([board])
    tensors = evaluator.model(tensors)

    malformed = tensors.clone()
    malformed["network", "policy_logits"] = torch.zeros((1, 1857))
    with pytest.raises(ValueError, match="shape"):
        evaluator.finish([board], malformed)

    wrong_mask = tensors.clone()
    wrong_mask["input", "legal_mask"].zero_()
    with pytest.raises(ValueError, match="does not match"):
        evaluator.finish([board], wrong_mask)

    wrong_planes = tensors.clone()
    wrong_planes["input", "planes"][0, 0, 0, 0] = 1 - wrong_planes["input", "planes"][0, 0, 0, 0]
    with pytest.raises(ValueError, match="input/planes does not match"):
        evaluator.finish([board], wrong_planes)

    missing = TensorDict(
        {
            ("input", "planes"): tensors["input", "planes"],
            ("input", "legal_mask"): tensors["input", "legal_mask"],
        },
        batch_size=[1],
    )
    with pytest.raises(ValueError, match="policy_logits"):
        evaluator.finish([board], missing)


def test_policy_view_reflects_explicit_runtime_tensor_changes():
    evaluation = fixture_evaluator().evaluate(chess.Board())
    d4 = evaluation.policy["d2d4"].index
    evaluation.tensors["network", "policy_logits"][d4] = 10
    evaluation.tensors["evaluation", "policy"] = torch.softmax(
        evaluation.tensors["network", "policy_logits"].masked_fill(
            ~evaluation.tensors["input", "legal_mask"], float("-inf")
        ),
        dim=0,
    )

    assert evaluation.policy.best_move == chess.Move.from_uci("d2d4")


def test_finish_rejects_non_floating_inputs_and_outputs():
    evaluator = fixture_evaluator()
    board = chess.Board()
    tensors = evaluator.model(evaluator.prepare([board]))

    integer_planes = tensors.clone()
    integer_planes["input", "planes"] = integer_planes["input", "planes"].to(torch.int64)
    with pytest.raises(ValueError, match="floating-point"):
        evaluator.finish([board], integer_planes)

    integer_policy = tensors.clone()
    integer_policy["network", "policy_logits"] = integer_policy["network", "policy_logits"].to(torch.int64)
    with pytest.raises(ValueError, match="floating-point"):
        evaluator.finish([board], integer_policy)


def test_evaluator_constructor_and_board_collection_validation():
    with pytest.raises(TypeError, match="LczeroModel"):
        LczeroEvaluator(nn.Identity())
    with pytest.raises(TypeError, match="InputFormat"):
        LczeroEvaluator(LczeroModel(PolicyOnlyNetwork(), out_keys=["policy"]), input_format="classical")
    with pytest.raises(TypeError, match="provenance"):
        LczeroEvaluator(LczeroModel(PolicyOnlyNetwork(), out_keys=["policy"]), provenance="fixture")
    with pytest.raises(ValueError, match="requires a policy head"):
        LczeroEvaluator(LczeroModel(nn.Identity(), out_keys=["value"]))
    with pytest.raises(ValueError, match="supports only"):
        LczeroEvaluator(LczeroModel(nn.Identity(), out_keys=["policy", "other"]))

    evaluator = fixture_evaluator()
    with pytest.raises(TypeError, match="sequence"):
        evaluator.prepare(chess.Board())
    with pytest.raises(ValueError, match="at least one"):
        evaluator.prepare([])
    with pytest.raises(TypeError, match="Every position"):
        evaluator.prepare([object()])


def test_from_path_forwards_model_and_evaluator_options(monkeypatch):
    model = LczeroModel(PolicyOnlyNetwork(), out_keys=["policy"])
    observed = {}

    def fake_from_path(path, **kwargs):
        observed["path"] = path
        observed["kwargs"] = kwargs
        return model

    monkeypatch.setattr(LczeroModel, "from_path", fake_from_path)

    evaluator = LczeroEvaluator.from_path("network.pt", weights_only=True)

    assert evaluator.model is model
    assert observed == {"path": "network.pt", "kwargs": {"weights_only": True}}


def test_evaluate_accepts_iterables_and_policy_only_models():
    evaluator = LczeroEvaluator(LczeroModel(PolicyOnlyNetwork(), out_keys=["policy"]))

    evaluations = evaluator.evaluate(board for board in (chess.Board(), chess.Board()))

    assert isinstance(evaluations, EvaluationBatch)
    assert len(evaluations) == 2
    assert evaluations[0].value is None


def test_evaluation_views_reject_incompatible_batch_shapes():
    evaluator = fixture_evaluator()
    tensors = evaluator.model(evaluator.prepare([chess.Board()]))

    with pytest.raises(ValueError, match="unbatched"):
        Evaluation(chess.Board(), tensors, evaluator.provenance, evaluator.input_format.value)
    with pytest.raises(ValueError, match="Board count"):
        EvaluationBatch([], tensors, evaluator.provenance, evaluator.input_format.value)
    row = tensors[0]
    with pytest.raises(TypeError, match="EvaluationProvenance"):
        Evaluation(chess.Board(), row, "not provenance", evaluator.input_format.value)
    with pytest.raises(ValueError, match="input format"):
        Evaluation(chess.Board(), row, evaluator.provenance, "")
    with pytest.raises(TypeError, match="EvaluationProvenance"):
        EvaluationBatch([chess.Board()], tensors, "not provenance", evaluator.input_format.value)
    with pytest.raises(ValueError, match="input format"):
        EvaluationBatch([chess.Board()], tensors, evaluator.provenance, "")


def test_finish_rejects_container_dtype_and_finite_value_errors():
    evaluator = fixture_evaluator()
    board = chess.Board()
    tensors = evaluator.model(evaluator.prepare([board]))

    with pytest.raises(TypeError, match="TensorDictBase"):
        evaluator.finish([board], {})
    with pytest.raises(ValueError, match="batch dimension"):
        evaluator.finish([board, board], tensors)

    wrong_mask_dtype = tensors.clone()
    wrong_mask_dtype["input", "legal_mask"] = wrong_mask_dtype["input", "legal_mask"].float()
    with pytest.raises(ValueError, match="dtype"):
        evaluator.finish([board], wrong_mask_dtype)

    nonfinite_planes = tensors.clone()
    nonfinite_planes["input", "planes"][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        evaluator.finish([board], nonfinite_planes)

    nonfinite_policy = tensors.clone()
    nonfinite_policy["network", "policy_logits"][0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        evaluator.finish([board], nonfinite_policy)


@pytest.mark.parametrize(
    "wdl",
    [
        [-0.1, 0.5, 0.6],
        [1.1, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ],
)
def test_finish_rejects_invalid_wdl_probabilities(wdl):
    evaluator = fixture_evaluator(wdl=True)
    board = chess.Board()
    tensors = evaluator.model(evaluator.prepare([board]))
    tensors["network", "wdl"] = torch.tensor([wdl])

    with pytest.raises(ValueError, match="probabilities"):
        evaluator.finish([board], tensors)


@pytest.mark.parametrize("value", [float("nan"), 1.1])
def test_finish_rejects_invalid_native_values(value):
    evaluator = fixture_evaluator()
    board = chess.Board()
    tensors = evaluator.model(evaluator.prepare([board]))
    tensors["network", "value"] = torch.tensor([[value]])

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        evaluator.finish([board], tensors)


def test_finish_validates_and_exposes_mlh():
    evaluator = fixture_evaluator()
    board = chess.Board()
    tensors = evaluator.model(evaluator.prepare([board]))
    tensors["network", "value"] = tensors["network", "value"].unsqueeze(-1)
    tensors["network", "mlh"] = torch.tensor([12.0])

    evaluation = evaluator.finish([board], tensors)[0]

    assert evaluation.mlh == pytest.approx(12)

    malformed = tensors.clone()
    malformed["network", "mlh"] = torch.ones((1, 2))
    with pytest.raises(ValueError, match="network/mlh must have shape"):
        evaluator.finish([board], malformed)

    nonfinite = tensors.clone()
    nonfinite["network", "mlh"] = torch.tensor([float("inf")])
    with pytest.raises(ValueError, match="network/mlh must be finite"):
        evaluator.finish([board], nonfinite)
