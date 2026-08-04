"""Contract tests for the TensorDict-centered Lczero evaluator."""

import chess
import pytest
import torch
from tensordict import TensorDict
from torch import nn

from lczerolens import LczeroModel
from lczerolens._codec import encode_move
from lczerolens.evaluation import ValueOrigin
from lczerolens.evaluator import LczeroEvaluator


class FixtureNetwork(nn.Module):
    def __init__(self, *, wdl: bool = False):
        super().__init__()
        self.wdl = wdl

    def forward(self, planes):
        batch = planes.shape[0]
        policy = torch.zeros((batch, 1858), device=planes.device)
        policy[:, encode_move(chess.Move.from_uci("e2e4"), chess.WHITE)] = 4
        policy[:, encode_move(chess.Move.from_uci("d2d4"), chess.WHITE)] = 2
        if self.wdl:
            return policy, torch.tensor([[0.6, 0.3, 0.1]], device=planes.device).repeat(batch, 1)
        return policy, torch.full((batch,), 0.25, device=planes.device)


def fixture_evaluator(*, wdl=False):
    heads = ["policy", "wdl" if wdl else "value"]
    return LczeroEvaluator(LczeroModel(FixtureNetwork(wdl=wdl), out_keys=heads))


def test_evaluate_one_position_has_natural_legal_policy_and_native_value():
    evaluation = fixture_evaluator().evaluate(chess.Board())

    assert evaluation.policy.best_move == chess.Move.from_uci("e2e4")
    assert evaluation.policy["e2e4"].probability > evaluation.policy["d2d4"].probability
    assert evaluation.policy["e2e4"].rank == 1
    assert evaluation.policy.top(2)[0].move == chess.Move.from_uci("e2e4")
    assert evaluation.value is not None
    assert evaluation.value.value == pytest.approx(0.25)
    assert evaluation.value.origin is ValueOrigin.NATIVE
    assert evaluation.wdl is None


def test_wdl_value_is_derived_without_overwriting_the_network_head():
    evaluation = fixture_evaluator(wdl=True).evaluate(chess.Board())

    assert evaluation.wdl is not None
    assert evaluation.wdl.win == pytest.approx(0.6)
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
    assert len(evaluations[1:]) == 1
    assert evaluations.tensors is tensors
    assert ("attr", "input", "planes") in tensors.keys(include_nested=True, leaves_only=True)
    assert torch.allclose(tensors["evaluation", "policy"].sum(-1), torch.ones(2))


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

    missing = TensorDict(
        {
            ("input", "planes"): tensors["input", "planes"],
            ("input", "legal_mask"): tensors["input", "legal_mask"],
        },
        batch_size=[1],
    )
    with pytest.raises(ValueError, match="policy_logits"):
        evaluator.finish([board], missing)
