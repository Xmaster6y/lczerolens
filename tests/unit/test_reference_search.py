"""Hermetic checks for the deterministic, replayable MCTS reference core."""

from dataclasses import replace

import pytest
import torch
from tensordict import TensorDict

from lczerolens import LczeroBoard
from lczerolens.reference_search import (
    ReferenceMCTS,
    SemanticReplayError,
    _Node,
    _apply_retained_root_delta,
    _retained_initial_root_state,
    _retained_root_transition,
    plan_retained_events,
    replay_retained_events,
    _replay_path,
    replay_root_events,
    replay_search_trace,
)
from lczerolens.search_trace import (
    BackupUpdate,
    EdgeStatistics,
    LeafRecord,
    PositionEvaluation,
    RootAction,
    SearchCapability,
    SearchProvenance,
    SimulationEvent,
    ValuePerspective,
    Wdl,
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


@pytest.mark.parametrize(
    ("fen", "simulations", "c_puct"),
    (
        ("8/8/8/8/8/8/4Q3/K1k5 w - - 0 1", 1, 0.0),  # terminal leaf
        ("7k/8/8/8/8/8/6r1/7K w - - 0 1", 3, 1.0),  # forced root move
        (None, 2, 0.0),  # repeated visit
        (None, 32, 1.5),  # competitive root
    ),
)
def test_semantic_replay_reconstructs_representative_search_results(fen, simulations, c_puct):
    board = LczeroBoard(fen) if fen is not None else LczeroBoard()
    trace = ReferenceMCTS(c_puct=c_puct).search(board, FixedEvaluator(), simulations)

    result = replay_search_trace(trace)

    assert result.root_statistics == tuple(action.statistics for action in trace.snapshots[-1].actions)
    assert sum(probability for _, probability in result.root_policy) == pytest.approx(1.0)
    assert result.selected_move == trace.snapshots[-1].selection.move


def test_retained_event_replay_supports_full_prefix_sparse_singleton_empty_and_complement_selections():
    trace = ReferenceMCTS(c_puct=1.5).search(LczeroBoard(), FixedEvaluator(), simulations=8)
    event_ids = tuple(event.event_id for event in trace.events)

    full = replay_retained_events(trace)
    prefix = replay_retained_events(trace, event_ids[:3])
    sparse = replay_retained_events(trace, event_ids[::2])
    singleton = replay_retained_events(trace, (event_ids[4],))
    empty = replay_retained_events(trace, ())
    complement = replay_retained_events(trace, event_ids[1::2])

    assert full.root_statistics == pytest.approx(replay_search_trace(trace).root_statistics)
    assert full.selected_move == trace.snapshots[-1].selection.move
    assert prefix.costs.simulations == 3
    assert sparse.costs.simulations == 4
    assert singleton.costs.simulations == 1
    assert empty.costs.simulations == 0
    assert sum(edge.visits or 0 for edge in empty.root_statistics) == 0
    assert tuple(event.event_id for event in sparse.plan.events) == event_ids[::2]
    assert [edge.visits for edge in full.root_statistics] == [
        (left.visits or 0) + (right.visits or 0)
        for left, right in zip(sparse.root_statistics, complement.root_statistics)
    ]


def test_retained_event_plan_preserves_trace_order_and_rejects_ambiguous_selections():
    trace = ReferenceMCTS().search(LczeroBoard(), FixedEvaluator(), simulations=3)
    first, second, third = (event.event_id for event in trace.events)

    plan = plan_retained_events(trace, (third, first))

    assert plan.retained_event_ids == (first, third)
    with pytest.raises(ValueError, match="must be unique"):
        plan_retained_events(trace, (first, first))
    with pytest.raises(ValueError, match="Unknown retained"):
        plan_retained_events(trace, (second, "missing"))


def test_full_retained_replay_is_not_limited_to_reference_mcts_provenance():
    trace = ReferenceMCTS().search(LczeroBoard(), FixedEvaluator(), simulations=2)
    other_provider = replace(trace, provenance=SearchProvenance(source="external-search", engine="external"))

    result = replay_retained_events(other_provider)

    assert result.root_statistics == tuple(action.statistics for action in other_provider.snapshots[-1].actions)
    assert result.selected_move == other_provider.snapshots[-1].selection.move


def test_retained_event_replay_clears_provider_specific_exploration_evidence():
    trace = ReferenceMCTS().search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    before = tuple(replace(edge, exploration=0.5) for edge in event.root_before)
    after = tuple(replace(edge, exploration=0.5) for edge in event.root_after)
    before_by_move = {edge.move: edge for edge in before}
    after_by_move = {edge.move: edge for edge in after}
    root_backup = event.backups[0]
    explored_event = replace(
        event,
        root_before=before,
        root_after=after,
        backups=(
            replace(
                root_backup,
                before=before_by_move[root_backup.before.move],
                after=after_by_move[root_backup.after.move],
            ),
        ),
    )
    snapshot = replace(trace.snapshots[-1], actions=tuple(RootAction(edge) for edge in after))
    explored_trace = replace(trace, events=(explored_event,), snapshots=(snapshot,))

    full = replay_retained_events(explored_trace)
    empty = replay_retained_events(explored_trace, ())

    assert all(edge.exploration is None for edge in full.root_statistics)
    assert all(edge.exploration is None for edge in empty.root_statistics)


def test_retained_event_replay_rejects_missing_or_malformed_retained_evidence():
    trace = ReferenceMCTS().search(LczeroBoard(), FixedEvaluator(), simulations=2)
    event = trace.events[0]
    before = event.root_before
    after = event.root_after
    assert before is not None and after is not None

    with pytest.raises(ValueError, match="at least one"):
        replay_retained_events(replace(trace, events=()))
    with pytest.raises(ValueError, match="non-empty initial"):
        _retained_initial_root_state(replace(event, root_before=()))
    with pytest.raises(ValueError, match="duplicate root"):
        _retained_initial_root_state(replace(event, root_before=(before[0], before[0])))
    with pytest.raises(ValueError, match="duplicate root"):
        _retained_root_transition(replace(event, root_before=(before[0], before[0])), set())
    with pytest.raises(ValueError, match="changes the root move set"):
        _retained_root_transition(replace(event, root_after=after[:-1]), {edge.move for edge in before})
    with pytest.raises(ValueError, match="exactly one root edge"):
        _retained_root_transition(replace(event, root_after=before), {edge.move for edge in before})
    with pytest.raises(ValueError, match="no matching root backup"):
        _retained_root_transition(replace(event, backups=()), {edge.move for edge in before})


def test_retained_root_delta_rejects_incompatible_or_incomplete_updates():
    trace = ReferenceMCTS().search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    before = event.root_before
    after = event.root_after
    assert before is not None and after is not None
    after_by_move = {edge.move: edge for edge in after}
    move = next(edge.move for edge in before if edge != after_by_move[edge.move])
    event_before = next(edge for edge in before if edge.move == move)
    event_after = next(edge for edge in after if edge.move == move)

    with pytest.raises(ValueError, match="incompatible"):
        _apply_retained_root_delta(before[1], event_before, event_after, event)
    with pytest.raises(ValueError, match="changes a root prior"):
        _apply_retained_root_delta(event_before, event_before, replace(event_after, prior=1.0), event)
    with pytest.raises(ValueError, match="root visit and value"):
        _apply_retained_root_delta(event_before, replace(event_before, visits=None), event_after, event)
    with pytest.raises(ValueError, match="root visits negative"):
        _apply_retained_root_delta(
            event_before,
            replace(event_before, visits=1, total_value=0.0, mean_value=0.0),
            replace(event_after, visits=0, total_value=0.0, mean_value=0.0),
            event,
        )


def test_semantic_replay_preserves_root_history_for_fivefold_repetition():
    board = LczeroBoard("8/8/6r1/8/6R1/8/K6k/8 b - - 0 1")
    cycle = ("h2h3", "a2a1", "h3h2", "a1a2")
    for _ in range(3):
        for move in cycle:
            board.push_uci(move)
    for move in cycle[:3]:
        board.push_uci(move)

    trace = ReferenceMCTS(c_puct=0.0).search(board, FixedEvaluator(), simulations=1)
    result = replay_search_trace(trace)

    assert trace.root_start_fen == "8/8/6r1/8/6R1/8/K6k/8 b - - 0 1"
    assert trace.root_move_history == tuple(board_move.uci() for board_move in board.move_stack)
    assert trace.events[0].leaf.terminal
    assert result.selected_move == trace.snapshots[-1].selection.move


def test_semantic_replay_requires_complete_and_consistent_root_history():
    trace = ReferenceMCTS().search(LczeroBoard(), FixedEvaluator(), simulations=1)

    with pytest.raises(ValueError, match="both a starting FEN and move sequence"):
        replace(trace, root_start_fen=None)
    with pytest.raises(ValueError, match="root history must reconstruct root_fen"):
        replace(trace, root_move_history=("e2e4",))
    with pytest.raises(ValueError, match="root history must be a legal sequence"):
        replace(trace, root_move_history=("e2e5",))

    legacy = replace(trace, root_start_fen=None, root_move_history=None)
    with pytest.raises(SemanticReplayError, match="root_history: reference traces need root history"):
        replay_search_trace(legacy)

    object.__setattr__(trace, "root_move_history", ("e2e4",))
    with pytest.raises(SemanticReplayError, match="root_history: root history does not reconstruct"):
        replay_search_trace(trace)


def test_semantic_replay_path_rejects_unreachable_internal_states():
    trace = ReferenceMCTS(c_puct=0.0).search(LczeroBoard(), FixedEvaluator(-0.4), simulations=2)
    first, second = trace.events
    unexpanded_root = _Node(LczeroBoard(), "node-0")

    with pytest.raises(SemanticReplayError, match="path continues beyond unexpanded"):
        _replay_path(unexpanded_root, {"node-0": unexpanded_root}, first, c_puct=0.0)
    with pytest.raises(SemanticReplayError, match="simulation path is empty"):
        _replay_path(unexpanded_root, {"node-0": unexpanded_root}, replace(first, path=()), c_puct=0.0)
    with pytest.raises(SemanticReplayError, match="path ends early at already expanded"):
        replay_search_trace(replace(trace, events=(first, replace(second, path=second.path[:1]))))


def test_semantic_replay_reports_first_path_divergence():
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=2)
    event = trace.events[0]
    wrong_move = next(edge.move for edge in event.root_before if edge.move != event.path[0].move)
    wrong_step = replace(event.path[0], move=wrong_move)
    invalid = replace(trace, events=(replace(event, path=(wrong_step,)), *trace.events[1:]))

    with pytest.raises(SemanticReplayError, match=r"Event simulation-0 path: depth 0 expected"):
        replay_search_trace(invalid)


def test_semantic_replay_reports_first_expansion_divergence():
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    evaluator = event.leaf.evaluator
    altered_logits = ((evaluator.legal_policy_logits[0][0], 100.0), *evaluator.legal_policy_logits[1:])
    invalid_leaf = replace(event.leaf, evaluator=replace(evaluator, legal_policy_logits=altered_logits))
    invalid = replace(trace, events=(replace(event, leaf=invalid_leaf),))

    with pytest.raises(SemanticReplayError, match=r"Event simulation-0 expansion: edge"):
        replay_search_trace(invalid)


def test_semantic_replay_reports_first_perspective_divergence():
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    wrong_evaluation = replace(event.leaf.evaluation, perspective=ValuePerspective.ROOT_PLAYER)
    invalid = replace(trace, events=(replace(event, leaf=replace(event.leaf, evaluation=wrong_evaluation)),))

    with pytest.raises(SemanticReplayError, match=r"Event simulation-0 leaf: expected side_to_move"):
        replay_search_trace(invalid)


def test_semantic_replay_rejects_recorded_post_state_instead_of_returning_it():
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    backup = event.backups[-1]
    altered_after = replace(
        backup.after,
        total_value=-backup.after.total_value,
        mean_value=-backup.after.mean_value,
    )
    altered_backup = replace(backup, signed_value=-backup.signed_value, after=altered_after)
    altered_root_after = tuple(altered_after if edge.move == altered_after.move else edge for edge in event.root_after)
    altered_actions = tuple(
        replace(action, statistics=altered_after) if action.statistics.move == altered_after.move else action
        for action in trace.snapshots[-1].actions
    )
    invalid = replace(
        trace,
        events=(replace(event, backups=(altered_backup,), root_after=altered_root_after),),
        snapshots=(replace(trace.snapshots[-1], actions=altered_actions),),
    )

    with pytest.raises(SemanticReplayError, match=r"Event simulation-0 backup: backup 0"):
        replay_search_trace(invalid)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda trace: replace(trace, provenance=replace(trace.provenance, source="other")), "provenance: semantic"),
        (
            lambda trace: replace(
                trace, provenance=replace(trace.provenance, parameters=trace.provenance.parameters[1:])
            ),
            "provenance: provenance needs",
        ),
        (
            lambda trace: replace(
                trace,
                provenance=replace(
                    trace.provenance,
                    parameters=(replace(trace.provenance.parameters[0], value=-1.0), *trace.provenance.parameters[1:]),
                ),
            ),
            "provenance: c_puct",
        ),
        (lambda trace: replace(trace, events=()), "events: at least one"),
        (
            lambda trace: replace(trace, events=(replace(trace.events[0], path=()),)),
            "Event simulation-0 path: a non-terminal root simulation needs a path",
        ),
        (
            lambda trace: replace(trace, events=(replace(trace.events[0], simulation=1),)),
            "Event simulation-0 sequence: expected simulation 0",
        ),
        (lambda trace: replace(trace, root_expansion=None), "root_expansion: reference trace needs"),
        (
            lambda trace: replace(
                trace,
                root_evaluator=replace(
                    trace.root_evaluator,
                    legal_policy_logits=trace.root_evaluator.legal_policy_logits[:-1],
                ),
            ),
            "root_expansion: expanded root edges",
        ),
        (
            lambda trace: replace(
                trace,
                root_expansion=replace(
                    trace.root_expansion,
                    edges=(
                        replace(trace.root_expansion.edges[0], perspective=ValuePerspective.SIDE_TO_MOVE),
                        *trace.root_expansion.edges[1:],
                    ),
                ),
            ),
            "root_expansion: root edge",
        ),
        (
            lambda trace: replace(
                trace,
                root_evaluator=replace(trace.root_evaluator, dtype="int64"),
            ),
            "expansion: unsupported evaluator dtype",
        ),
    ),
)
def test_semantic_replay_rejects_invalid_trace_level_evidence(mutation, message):
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)

    with pytest.raises(SemanticReplayError, match=message):
        replay_search_trace(mutation(trace))


@pytest.mark.parametrize(
    ("mutate_event", "message"),
    (
        (
            lambda event, trace: replace(event, leaf=replace(event.leaf, node_id="wrong-node")),
            "Event simulation-0 leaf: path ends",
        ),
        (
            lambda event, trace: replace(event, leaf=replace(event.leaf, terminal=True)),
            "Event simulation-0 leaf: terminal flag",
        ),
        (
            lambda event, trace: replace(event, expansion=None),
            "Event simulation-0 expansion: non-terminal first visits",
        ),
        (
            lambda event, trace: replace(event, expansion=replace(event.expansion, node_id="wrong-node")),
            "Event simulation-0 expansion: expansion node",
        ),
        (
            lambda event, trace: replace(
                event,
                leaf=replace(
                    event.leaf,
                    evaluator=replace(
                        event.leaf.evaluator,
                        legal_policy_logits=event.leaf.evaluator.legal_policy_logits[:-1],
                    ),
                ),
            ),
            "Event simulation-0 expansion: expanded edges",
        ),
        (
            lambda event, trace: replace(
                event,
                expansion=replace(
                    event.expansion,
                    edges=(
                        replace(event.expansion.edges[0], perspective=ValuePerspective.ROOT_PLAYER),
                        *event.expansion.edges[1:],
                    ),
                ),
            ),
            "Event simulation-0 expansion: edge",
        ),
        (
            lambda event, trace: replace(event, backups=(*event.backups, event.backups[0])),
            "Event simulation-0 backup: expected 1 backups, got 2",
        ),
    ),
)
def test_semantic_replay_rejects_invalid_event_evidence(mutate_event, message):
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    invalid_event = mutate_event(trace.events[0], trace)
    invalid = replace(trace, events=(invalid_event,))

    with pytest.raises(SemanticReplayError, match=message):
        replay_search_trace(invalid)


def test_semantic_replay_rejects_terminal_evaluator_and_value_divergences():
    board = LczeroBoard("8/8/8/8/8/8/4Q3/K1k5 w - - 0 1")
    trace = ReferenceMCTS(c_puct=0.0).search(board, FixedEvaluator(), simulations=1)
    event = trace.events[0]
    with_evaluator = replace(event, leaf=replace(event.leaf, evaluator=trace.root_evaluator))
    wrong_value = replace(
        event,
        leaf=replace(event.leaf, evaluation=replace(event.leaf.evaluation, value=0.5)),
    )

    with pytest.raises(SemanticReplayError, match="terminal leaves cannot have"):
        replay_search_trace(replace(trace, events=(with_evaluator,)))
    with pytest.raises(SemanticReplayError, match="terminal value should be"):
        replay_search_trace(replace(trace, events=(wrong_value,)))


def test_semantic_replay_rejects_reused_or_changed_child_identity():
    trace = ReferenceMCTS(c_puct=0.0).search(LczeroBoard(), FixedEvaluator(-0.4), simulations=2)
    first, second = trace.events
    reused_root = replace(first.path[0], child_id=first.path[0].node_id)
    changed_child = replace(second.path[0], child_id="other-child")

    with pytest.raises(SemanticReplayError, match="new child ID .* already in use"):
        replay_search_trace(replace(trace, events=(replace(first, path=(reused_root,)), second)))
    with pytest.raises(SemanticReplayError, match="edge .* points to"):
        replay_search_trace(replace(trace, events=(first, replace(second, path=(changed_child, *second.path[1:])))))


def test_semantic_replay_rejects_wrong_selected_move():
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    final = trace.snapshots[-1]
    other_move = next(
        action.statistics.move for action in final.actions if action.statistics.move != final.selection.move
    )
    invalid_selection = replace(final.selection, move=other_move)

    with pytest.raises(SemanticReplayError, match="result: selected move diverges"):
        replay_search_trace(replace(trace, snapshots=(replace(final, selection=invalid_selection),)))


@pytest.mark.parametrize(
    ("root_before", "message"),
    (
        (lambda edges: edges[:-1], "initial root edges do not match"),
        (
            lambda edges: (
                replace(edges[0], perspective=ValuePerspective.SIDE_TO_MOVE),
                *edges[1:],
            ),
            "root edge .* has perspective",
        ),
        (
            lambda edges: (replace(edges[0], visits=1, total_value=0.0, mean_value=0.0), *edges[1:]),
            "root edge .* is not an unvisited",
        ),
    ),
)
def test_semantic_replay_rejects_corrupt_initial_root_state(root_before, message):
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    # Model corruption that can arrive from untrusted persisted trace data.
    object.__setattr__(event, "root_before", root_before(event.root_before))

    with pytest.raises(SemanticReplayError, match=message):
        replay_search_trace(trace)


def test_semantic_replay_rejects_path_past_first_unexpanded_node():
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    extra = replace(event.path[0], node_id=event.path[0].child_id, child_id="extra-child")

    with pytest.raises(SemanticReplayError, match="path continues after first unexpanded node"):
        replay_search_trace(replace(trace, events=(replace(event, path=(*event.path, extra)),)))


def test_semantic_replay_requires_scalar_leaf_value_for_backup():
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    perspective = event.leaf.evaluation.perspective
    wdl_only = PositionEvaluation(perspective, wdl=Wdl(0.5, 0.25, 0.25, perspective))
    invalid_leaf = replace(event.leaf, evaluation=wdl_only)

    with pytest.raises(SemanticReplayError, match="backup: scalar leaf value is required"):
        replay_search_trace(replace(trace, events=(replace(event, leaf=invalid_leaf),)))


def test_semantic_replay_rejects_backup_post_state_divergence():
    trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    event = trace.events[0]
    backup = event.backups[0]
    altered_after = replace(backup.after, exploration=0.1)
    altered_backup = replace(backup, after=altered_after)
    altered_root_after = tuple(altered_after if edge.move == altered_after.move else edge for edge in event.root_after)
    altered_actions = tuple(
        replace(action, statistics=altered_after) if action.statistics.move == altered_after.move else action
        for action in trace.snapshots[-1].actions
    )
    invalid = replace(
        trace,
        events=(replace(event, backups=(altered_backup,), root_after=altered_root_after),),
        snapshots=(replace(trace.snapshots[-1], actions=altered_actions),),
    )

    with pytest.raises(SemanticReplayError, match="backup 0 post-state diverges"):
        replay_search_trace(invalid)


def test_semantic_replay_rejects_root_and_final_action_state_divergence():
    root_trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=2)
    object.__setattr__(root_trace.events[1], "root_before", None)
    with pytest.raises(SemanticReplayError, match="root_before: recorded root state diverges"):
        replay_search_trace(root_trace)

    result_trace = ReferenceMCTS(c_puct=1.0).search(LczeroBoard(), FixedEvaluator(), simulations=1)
    final = result_trace.snapshots[-1]
    object.__setattr__(final, "actions", final.actions[:-1])
    with pytest.raises(SemanticReplayError, match="result: final root action statistics diverge"):
        replay_search_trace(result_trace)


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
