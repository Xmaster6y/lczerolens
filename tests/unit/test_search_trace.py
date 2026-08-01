"""Contracts for engine-independent search trace records."""

import math

import pytest

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


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SearchParameter("", 1), "parameter names"),
        (lambda: SearchParameter("temperature", math.nan), "must be finite"),
        (lambda: SearchProvenance(source=""), "source must not be empty"),
        (
            lambda: SearchProvenance(
                source="reference", parameters=(SearchParameter("seed", 1), SearchParameter("seed", 2))
            ),
            "must be unique",
        ),
        (lambda: SearchBudget(SearchBudgetUnit.NODES), "requested or observed"),
        (lambda: SearchBudget(SearchBudgetUnit.NODES, requested=-1), "non-negative"),
        (lambda: Wdl(-0.1, 0.5, 0.6, ValuePerspective.ROOT_PLAYER), "probabilities"),
        (lambda: Wdl(0.2, 0.2, 0.2, ValuePerspective.ROOT_PLAYER), "sum to one"),
        (lambda: Wdl(0.5, 0.0, 0.5, ValuePerspective.ROOT_PLAYER).scalar(math.nan), "draw_score"),
        (lambda: PositionEvaluation(ValuePerspective.ROOT_PLAYER), "scalar value or WDL"),
        (lambda: PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=2.0), r"in \[-1, 1\]"),
        (
            lambda: PositionEvaluation(
                ValuePerspective.ROOT_PLAYER,
                wdl=Wdl(0.5, 0.0, 0.5, ValuePerspective.WHITE),
            ),
            "perspectives must agree",
        ),
        (lambda: PrincipalVariation(()), "must contain"),
        (lambda: EdgeStatistics("", ValuePerspective.ROOT_PLAYER), "non-empty"),
        (lambda: EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, prior=2.0), "P must"),
        (lambda: EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=1.5), "N must"),
        (lambda: EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, total_value=math.nan), "W must"),
        (lambda: EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, mean_value=2.0), "Q must"),
        (lambda: EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, exploration=-0.1), "U must"),
        (
            lambda: EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=1, total_value=1.1),
            "W must be in",
        ),
        (lambda: RootSelection("", "bestmove", "engine-defined"), "must be explicit"),
        (lambda: RootSelection("e2e4", "bestmove", "engine-defined", temperature=-1), "non-negative"),
        (
            lambda: RootSnapshot(sequence=-1, evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0)),
            "non-negative",
        ),
        (lambda: RootSnapshot(sequence=0), "selection, evaluation, or exposed actions"),
    ],
)
def test_record_validation_errors(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_root_and_event_record_validation_errors():
    edge = EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, prior=1.0, visits=0)
    with pytest.raises(ValueError, match="must start"):
        RootAction(edge, principal_variation=PrincipalVariation(("d2d4",)))
    with pytest.raises(ValueError, match="unique moves"):
        RootSnapshot(
            sequence=0,
            actions=(RootAction(edge), RootAction(edge)),
        )
    with pytest.raises(ValueError, match="dtype and device"):
        EvaluatorCall("", "cpu", "cpu")
    with pytest.raises(ValueError, match="unique, non-empty"):
        EvaluatorCall("float32", "cpu", "cpu", legal_policy_logits=(("e2e4", 0.0), ("e2e4", 1.0)))
    with pytest.raises(ValueError, match="must be finite"):
        EvaluatorCall("float32", "cpu", "cpu", legal_policy_logits=(("e2e4", math.nan),))
    with pytest.raises(ValueError, match="unique moves"):
        NodeExpansion("root", (edge, edge))
    with pytest.raises(ValueError, match="explicit prior"):
        NodeExpansion("root", (EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER),))
    with pytest.raises(ValueError, match="sum to one"):
        NodeExpansion("root", (EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, prior=0.5),))


def test_optional_evaluator_and_empty_expansion_records_are_valid():
    EvaluatorCall("float32", "cpu", "cpu")
    EvaluatorCall("float32", "cpu", "cpu", legal_policy_logits=(("e2e4", 0.0),))
    NodeExpansion("terminal", ())


def test_backup_and_event_validation_errors():
    before = EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=0, total_value=0.0, mean_value=0.0)
    after = EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=1, total_value=0.5, mean_value=0.5)
    with pytest.raises(ValueError, match=r"in \[-1, 1\]"):
        BackupUpdate("root", 2.0, before, after)
    with pytest.raises(ValueError, match="same edge"):
        BackupUpdate(
            "root",
            0.5,
            before,
            EdgeStatistics("d2d4", ValuePerspective.ROOT_PLAYER, visits=1, total_value=0.5, mean_value=0.5),
        )
    with pytest.raises(ValueError, match="pre/post N, W, and Q"):
        BackupUpdate("root", 0.5, EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER), after)
    with pytest.raises(ValueError, match="N_post"):
        BackupUpdate(
            "root",
            0.5,
            before,
            EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER, visits=2, total_value=0.5, mean_value=0.25),
        )
    with pytest.raises(ValueError, match="ID and non-negative"):
        SimulationEvent(
            event_id="",
            simulation=0,
            path=(PathStep("root", "e2e4", "child"),),
            leaf=LeafRecord("child", PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0), False),
            backups=(),
        )


def test_search_trace_validation_errors():
    root_only = RootSnapshot(sequence=0, evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0))
    action_snapshot = RootSnapshot(
        sequence=0,
        budget=SearchBudget(SearchBudgetUnit.NODES, observed=0),
        actions=(),
    )
    trace_args = dict(
        root_fen=START_FEN,
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.ROOT_RESULT,
        provenance=SearchProvenance(source="test"),
        snapshots=(root_only,),
    )
    with pytest.raises(ValueError, match="root_player must"):
        SearchTrace(**(trace_args | {"root_player": "white"}))
    with pytest.raises(ValueError, match="valid chess FEN"):
        SearchTrace(**(trace_args | {"root_fen": "not a FEN"}))
    with pytest.raises(ValueError, match="must match"):
        SearchTrace(**(trace_args | {"root_fen": START_FEN.replace(" w ", " b ")}))
    with pytest.raises(ValueError, match="must contain at least one"):
        SearchTrace(**(trace_args | {"snapshots": ()}))
    with pytest.raises(ValueError, match="unique and increasing"):
        SearchTrace(**(trace_args | {"snapshots": (root_only, root_only)}))
    with pytest.raises(ValueError, match="Full-event capability"):
        SearchTrace(
            **(
                trace_args
                | {
                    "capability": SearchCapability.FULL_EVENTS,
                    "snapshots": (action_snapshot,),
                }
            )
        )

    with pytest.raises(ValueError, match="Every root action"):
        SearchTrace(
            **(
                trace_args
                | {
                    "snapshots": (
                        RootSnapshot(
                            sequence=0,
                            actions=(RootAction(EdgeStatistics("e2e5", ValuePerspective.ROOT_PLAYER)),),
                        ),
                    )
                }
            )
        )
    for variation, message in (
        (("e2e4", "not-a-move"), "valid UCI"),
        (("e2e4", "e2e4"), "legal in sequence"),
    ):
        with pytest.raises(ValueError, match=message):
            SearchTrace(
                **(
                    trace_args
                    | {
                        "snapshots": (
                            RootSnapshot(
                                sequence=0,
                                actions=(
                                    RootAction(
                                        EdgeStatistics("e2e4", ValuePerspective.ROOT_PLAYER),
                                        principal_variation=PrincipalVariation(variation),
                                    ),
                                ),
                            ),
                        )
                    }
                )
            )

    event = SimulationEvent(
        event_id="event-0",
        simulation=0,
        path=(),
        leaf=LeafRecord("root", PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0), False),
        backups=(),
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        SearchTrace(**(trace_args | {"events": (event, event)}))
    repeated_simulation = SimulationEvent(
        event_id="event-1",
        simulation=0,
        path=(),
        leaf=event.leaf,
        backups=(),
    )
    with pytest.raises(ValueError, match="indices must be unique"):
        SearchTrace(**(trace_args | {"events": (event, repeated_simulation)}))
