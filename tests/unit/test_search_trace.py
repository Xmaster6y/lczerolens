"""Contracts for engine-independent search trace records."""

import pytest

from lczerolens.search_trace import (
    BackupUpdate,
    ChessPlayer,
    EdgeStatistics,
    LeafRecord,
    PositionEvaluation,
    PrincipalVariation,
    RootAction,
    RootSelection,
    RootSnapshot,
    SearchBudget,
    SearchBudgetUnit,
    SearchCapability,
    SearchCapabilityError,
    SearchParameter,
    SearchProvenance,
    SearchTrace,
    SimulationEvent,
    ValuePerspective,
    Wdl,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_current_python_search_fits_root_action_schema():
    """The legacy Python MCTS exposes P, N, and Q but need not invent W or U."""
    trace = SearchTrace(
        root_fen=START_FEN,
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.ROOT_ACTION_STATS,
        provenance=SearchProvenance(
            source="lczerolens-reference",
            engine="legacy-python-mcts",
            parameters=(SearchParameter("c_puct", 1.0),),
        ),
        snapshots=(
            RootSnapshot(
                sequence=0,
                budget=SearchBudget(SearchBudgetUnit.SIMULATIONS, requested=8, observed=8),
                selection=RootSelection("e2e4", rule="maximum N", tie_break="legal move order"),
                actions=(
                    RootAction(
                        EdgeStatistics(
                            move="e2e4",
                            perspective=ValuePerspective.ROOT_PLAYER,
                            prior=0.6,
                            visits=5,
                            mean_value=0.2,
                        )
                    ),
                    RootAction(
                        EdgeStatistics(
                            move="d2d4",
                            perspective=ValuePerspective.ROOT_PLAYER,
                            prior=0.4,
                            visits=3,
                            mean_value=0.1,
                        )
                    ),
                ),
            ),
        ),
    )

    assert trace.supports(SearchCapability.ROOT_ACTION_STATS)
    assert trace.snapshots[0].actions[0].statistics.total_value is None
    with pytest.raises(SearchCapabilityError, match="full_events"):
        trace.require(SearchCapability.FULL_EVENTS)


def test_captured_lc0_output_fits_budgeted_root_snapshot_schema():
    """An lc0 adapter can retain reported root data without claiming events."""
    trace = SearchTrace(
        root_fen=START_FEN,
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.ROOT_SNAPSHOTS,
        provenance=SearchProvenance(
            source="lc0-verbose-move-stats",
            engine="lc0",
            engine_version="v0.31.2",
            network="example.pb.gz",
            network_checksum="sha256:fixture",
            parameters=(SearchParameter("Threads", 1),),
        ),
        snapshots=(
            RootSnapshot(
                sequence=0,
                budget=SearchBudget(SearchBudgetUnit.NODES, requested=100, observed=100),
                evaluation=PositionEvaluation(
                    perspective=ValuePerspective.ROOT_PLAYER,
                    wdl=Wdl(0.45, 0.30, 0.25, ValuePerspective.ROOT_PLAYER),
                ),
                selection=RootSelection("e2e4", rule="engine bestmove", tie_break="engine-defined"),
                actions=(
                    RootAction(
                        EdgeStatistics(
                            move="e2e4",
                            perspective=ValuePerspective.ROOT_PLAYER,
                            prior=0.4,
                            visits=60,
                            mean_value=0.18,
                            exploration=0.03,
                        ),
                        evaluation=PositionEvaluation(
                            perspective=ValuePerspective.ROOT_PLAYER,
                            wdl=Wdl.from_win_loss_draw(0.2, 0.3, ValuePerspective.ROOT_PLAYER),
                        ),
                        leaf_evaluation=PositionEvaluation(
                            perspective=ValuePerspective.ROOT_PLAYER,
                            value=0.18,
                        ),
                        principal_variation=PrincipalVariation(("e2e4", "e7e5", "g1f3")),
                    ),
                    RootAction(
                        EdgeStatistics(
                            move="d2d4",
                            perspective=ValuePerspective.ROOT_PLAYER,
                            prior=0.3,
                            visits=40,
                            mean_value=0.15,
                            exploration=0.04,
                        )
                    ),
                ),
            ),
        ),
    )

    assert trace.require(SearchCapability.ROOT_ACTION_STATS) is trace
    assert trace.events is None
    assert trace.snapshots[0].evaluation.wdl.scalar() == pytest.approx(0.2)
    action_wdl = trace.snapshots[0].actions[0].evaluation.wdl
    assert (action_wdl.win, action_wdl.draw, action_wdl.loss) == pytest.approx((0.45, 0.3, 0.25))
    assert trace.snapshots[0].actions[0].leaf_evaluation.value == pytest.approx(0.18)


def test_root_only_trace_keeps_unavailable_actions_absent():
    trace = SearchTrace(
        root_fen=START_FEN,
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.ROOT_RESULT,
        provenance=SearchProvenance(source="uci"),
        snapshots=(
            RootSnapshot(
                sequence=0,
                selection=RootSelection("e2e4", rule="bestmove", tie_break="engine-defined"),
            ),
        ),
    )

    assert trace.snapshots[0].actions is None
    with pytest.raises(SearchCapabilityError, match="root_action_stats"):
        trace.require(SearchCapability.ROOT_ACTION_STATS)


def test_capability_claims_require_their_records():
    snapshot = RootSnapshot(
        sequence=0,
        selection=RootSelection("e2e4", rule="bestmove", tie_break="engine-defined"),
    )

    with pytest.raises(ValueError, match="requires actions"):
        SearchTrace(
            root_fen=START_FEN,
            root_player=ChessPlayer.WHITE,
            capability=SearchCapability.ROOT_ACTION_STATS,
            provenance=SearchProvenance(source="invalid"),
            snapshots=(snapshot,),
        )
    with pytest.raises(ValueError, match="requires a budget"):
        SearchTrace(
            root_fen=START_FEN,
            root_player=ChessPlayer.WHITE,
            capability=SearchCapability.ROOT_SNAPSHOTS,
            provenance=SearchProvenance(source="invalid"),
            snapshots=(
                RootSnapshot(
                    sequence=0,
                    evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0),
                    actions=(),
                ),
            ),
        )


def test_q_is_w_over_n_with_explicit_zero_visit_convention():
    EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=0, total_value=0.0, mean_value=0.0)
    EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=4, total_value=1.0, mean_value=0.25)

    with pytest.raises(ValueError, match="Q must equal W / N"):
        EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=4, total_value=1.0, mean_value=0.5)


def test_backup_update_enforces_replay_transition():
    before = EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=1, total_value=0.2, mean_value=0.2)
    after = EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=2, total_value=0.5, mean_value=0.25)

    BackupUpdate("root", signed_value=0.3, before=before, after=after)
    with pytest.raises(ValueError, match="W_post"):
        BackupUpdate(
            "root",
            signed_value=0.4,
            before=before,
            after=after,
        )


def test_selected_move_must_be_among_exposed_root_actions():
    with pytest.raises(ValueError, match="selected root move"):
        RootSnapshot(
            sequence=0,
            selection=RootSelection("e2e4", rule="maximum N", tie_break="UCI order"),
            actions=(RootAction(EdgeStatistics("d2d4", ValuePerspective.ROOT_PLAYER, visits=1)),),
        )


def test_trace_rejects_illegal_root_actions():
    with pytest.raises(ValueError, match="legal in root_fen"):
        SearchTrace(
            root_fen=START_FEN,
            root_player=ChessPlayer.WHITE,
            capability=SearchCapability.ROOT_RESULT,
            provenance=SearchProvenance(source="invalid"),
            snapshots=(
                RootSnapshot(
                    sequence=0,
                    selection=RootSelection("e2e5", rule="bestmove", tie_break="engine-defined"),
                ),
            ),
        )


def test_replayable_capability_requires_pre_and_post_root_state():
    before = EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=0, total_value=0.0, mean_value=0.0)
    after = EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=1, total_value=0.5, mean_value=0.5)
    backup = BackupUpdate("root", signed_value=0.5, before=before, after=after)
    event = SimulationEvent(
        event_id="simulation-0",
        simulation=0,
        path=(),
        leaf=LeafRecord(
            node_id="root",
            evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.5),
            terminal=False,
        ),
        backups=(backup,),
    )
    snapshot = RootSnapshot(
        sequence=0,
        budget=SearchBudget(SearchBudgetUnit.SIMULATIONS, observed=1),
        selection=RootSelection("e2e4", rule="maximum N", tie_break="UCI order"),
        actions=(RootAction(after),),
    )

    with pytest.raises(ValueError, match="Replayable capability"):
        SearchTrace(
            root_fen=START_FEN,
            root_player=ChessPlayer.WHITE,
            capability=SearchCapability.REPLAYABLE,
            provenance=SearchProvenance(source="reference"),
            snapshots=(snapshot,),
            events=(event,),
        )

    replayable_event = SimulationEvent(
        event_id=event.event_id,
        simulation=event.simulation,
        path=event.path,
        leaf=event.leaf,
        backups=event.backups,
        root_before=(before,),
        root_after=(after,),
    )
    trace = SearchTrace(
        root_fen=START_FEN,
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.REPLAYABLE,
        provenance=SearchProvenance(source="reference"),
        snapshots=(snapshot,),
        events=(replayable_event,),
    )
    assert trace.require(SearchCapability.REPLAYABLE) is trace
