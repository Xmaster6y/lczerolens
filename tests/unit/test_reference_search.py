"""Hermetic checks for the deterministic, replayable MCTS reference core."""

import pytest
import torch
from tensordict import TensorDict

from lczerolens import LczeroBoard
from lczerolens.reference_search import ReferenceMCTS, replay_root_events
from lczerolens.search_trace import (
    BackupUpdate,
    EdgeStatistics,
    LeafRecord,
    PositionEvaluation,
    SearchCapability,
    SimulationEvent,
    ValuePerspective,
)


class FixedEvaluator:
    """A deterministic full-policy evaluator matching the #130 model contract."""

    def __init__(self, value=0.25):
        self.value = value

    def __call__(self, board):
        policy = torch.zeros(1858, dtype=torch.float32)
        for rank, move in enumerate(sorted(board.legal_moves, key=lambda candidate: candidate.uci())):
            policy[board.encode_move(move, board.turn)] = float(rank)
        return TensorDict({"policy": policy, "value": torch.tensor(self.value)})


class BatchedFixedEvaluator(FixedEvaluator):
    """Match the singleton TensorDict batch emitted by LczeroModel."""

    def __call__(self, board):
        output = super().__call__(board)
        return TensorDict(
            {"policy": output["policy"].unsqueeze(0), "value": output["value"].reshape(1)}, batch_size=[1]
        )


@pytest.mark.parametrize("simulations", (1, 2, 8, 32))
def test_reference_search_is_deterministic_and_replayable(simulations):
    board = LczeroBoard()
    trace = ReferenceMCTS(c_puct=1.5).search(board, FixedEvaluator(), simulations)
    repeat = ReferenceMCTS(c_puct=1.5).search(board, FixedEvaluator(), simulations)

    assert trace.capability is SearchCapability.REPLAYABLE
    assert trace.events == repeat.events
    assert replay_root_events(trace.events) == tuple(action.statistics for action in trace.snapshots[-1].actions)
    if simulations == 1:
        assert trace.snapshots[-1].selection.move == "a2a3"  # stable UCI tie at zero visits
    for event in trace.events:
        assert event.replayable
        assert event.leaf.evaluator is not None
    for event in (event for event in trace.events if event.expansion is not None):
        assert sum(edge.prior for edge in event.expansion.edges) == pytest.approx(1.0)
        for backup in event.backups:
            assert backup.after.visits == backup.before.visits + 1
            assert backup.after.total_value == pytest.approx(backup.before.total_value + backup.signed_value)
            assert backup.after.mean_value == pytest.approx(backup.after.total_value / backup.after.visits)


def test_reference_search_revisits_a_child_and_alternates_backup_signs():
    board = LczeroBoard()
    trace = ReferenceMCTS(c_puct=0.0).search(board, FixedEvaluator(-0.4), simulations=2)

    assert len(trace.events[1].path) == 2
    assert [backup.signed_value for backup in trace.events[1].backups] == pytest.approx((0.4, -0.4))


def test_reference_search_accepts_the_canonical_singleton_evaluator_batch():
    trace = ReferenceMCTS().search(LczeroBoard(), BatchedFixedEvaluator(), simulations=1)

    assert trace.events[0].leaf.evaluator.legal_policy_logits


def test_reference_search_records_terminal_leaf_without_evaluator_call():
    board = LczeroBoard("8/8/8/8/8/8/4Q3/K1k5 w - - 0 1")
    trace = ReferenceMCTS(c_puct=0.0).search(board, FixedEvaluator(), simulations=1)

    assert trace.events[0].leaf.terminal
    assert trace.events[0].leaf.evaluator is None


def test_reference_search_rejects_invalid_evaluator_policy():
    def invalid(_board):
        return TensorDict({"policy": torch.full((1858,), float("nan")), "value": torch.tensor(0.0)})

    with pytest.raises(ValueError, match="non-finite"):
        ReferenceMCTS().search(LczeroBoard(), invalid, simulations=1)


def test_replayer_rejects_an_event_that_changes_the_root_move_set():
    before = EdgeStatistics("a2a3", ValuePerspective.ROOT_PLAYER, visits=0, total_value=0.0, mean_value=0.0)
    after = EdgeStatistics("a2a3", ValuePerspective.ROOT_PLAYER, visits=1, total_value=0.5, mean_value=0.5)
    extra = EdgeStatistics("a2a4", ValuePerspective.ROOT_PLAYER, visits=0, total_value=0.0, mean_value=0.0)
    event = SimulationEvent(
        "event-0",
        0,
        (),
        LeafRecord("leaf", PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.5), False),
        (BackupUpdate("root", 0.5, before, after),),
        root_before=(before,),
        root_after=(after, extra),
    )

    with pytest.raises(ValueError, match="root move set"):
        replay_root_events((event,))
