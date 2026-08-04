"""Synthetic positive, null, and failure cases for observable behaviour."""

from pathlib import Path

import chess
import pytest
import torch
from tensordict import TensorDict

from lczerolens import LczeroBoard
from lczerolens.behavior import (
    BehaviorMetric,
    ControlKind,
    METRIC_DEFINITIONS,
    compare_counterfactual_behavior,
    compare_evaluator_to_search,
    compare_search_decision,
    compare_search_events,
    evaluator_behavior,
)
from lczerolens.facts import FactPerspective, MaterialAnalyzer
from lczerolens.lc0_adapter import Lc0RootSnapshotParser, Lc0SearchRequest
from lczerolens.move_evidence import analyze_variation
from lczerolens.reference_search import ReferenceMCTS
from lczerolens.search_trace import (
    ChessPlayer,
    EdgeStatistics,
    PrincipalVariation,
    RootAction,
    RootSelection,
    RootSnapshot,
    SearchBudget,
    SearchBudgetUnit,
    SearchCapability,
    SearchCapabilityError,
    SearchProvenance,
    SearchTrace,
    ValuePerspective,
)


ROOT_FEN = chess.STARTING_FEN


def output_for(board, logits, *, value=None, wdl=None, mlh=None):
    policy = torch.full((1858,), -10.0)
    for move, logit in logits.items():
        policy[board.encode_move(chess.Move.from_uci(move), board.turn)] = logit
    values = {"policy": policy}
    if value is not None:
        values["value"] = torch.tensor(value)
    if wdl is not None:
        values["wdl"] = torch.tensor(wdl)
    if mlh is not None:
        values["mlh"] = torch.tensor(mlh)
    return TensorDict(values)


class FixedEvaluator:
    def __call__(self, board):
        ranked = {
            move.uci(): float(rank) for rank, move in enumerate(sorted(board.legal_moves, key=lambda item: item.uci()))
        }
        if board.fen() == ROOT_FEN:
            ranked |= {"e2e4": 30.0, "d2d4": 20.0}
        return output_for(board, ranked, value=0.2)


def test_evaluator_behavior_masks_illegal_moves_and_defines_optional_heads_and_ties():
    board = LczeroBoard()
    output = output_for(board, {"e2e4": 3.0, "d2d4": 3.0}, value=0.25, wdl=(0.5, 0.3, 0.2), mlh=17)
    illegal_index = next(index for index in range(1858) if index not in set(board.get_legal_indices().tolist()))
    output["policy"][illegal_index] = 100.0

    behavior = evaluator_behavior(board, output)

    assert behavior.selection_ties == ("d2d4", "e2e4")
    assert behavior.selected_move == "d2d4"
    assert behavior.action("e7e5") is None
    assert behavior.action("d2d4").rank == behavior.action("e2e4").rank == 1
    assert sum(action.probability for action in behavior.actions) == pytest.approx(1.0)
    assert behavior.evaluation.wdl.scalar() == pytest.approx(0.3)
    assert behavior.mlh == 17


def test_counterfactual_comparison_separates_positive_target_and_collateral_effects():
    board = LczeroBoard()
    original = evaluator_behavior(board, output_for(board, {"e2e4": 4.0, "d2d4": 2.0}))
    modified = evaluator_behavior(board, output_for(board, {"e2e4": 1.0, "d2d4": 4.0}))

    comparison = compare_counterfactual_behavior(original, modified, ("e2e4",), control_kind=ControlKind.MATCHED)

    assert comparison.targets[0].probability_delta < 0
    assert comparison.targets[0].rank_delta > 0
    assert "e2e4" not in comparison.collateral.moves
    assert comparison.collateral.probability_total_variation > 0


@pytest.mark.parametrize("kind", tuple(ControlKind))
def test_null_controls_are_first_class_structured_outputs(kind):
    board = LczeroBoard()
    behavior = evaluator_behavior(board, output_for(board, {"e2e4": 2.0}))

    comparison = compare_counterfactual_behavior(behavior, behavior, ("e2e4",), control_kind=kind)

    assert comparison.control_kind is kind
    assert comparison.targets[0].probability_delta == 0
    assert comparison.collateral.probability_total_variation == 0


def test_failures_are_explicit_for_missing_heads_bad_targets_and_capabilities():
    board = LczeroBoard()
    with pytest.raises(ValueError, match="missing required 'policy'"):
        evaluator_behavior(board, TensorDict({"value": torch.tensor(0.0)}))
    behavior = evaluator_behavior(board, output_for(board, {"e2e4": 1.0}))
    with pytest.raises(ValueError, match="target_moves"):
        compare_counterfactual_behavior(behavior, behavior, ())

    root_result = Lc0RootSnapshotParser().parse(
        ["bestmove e2e4"],
        request=Lc0SearchRequest(ROOT_FEN, nodes=1),
        engine_version="test",
        network="test",
    )
    with pytest.raises(SearchCapabilityError):
        compare_evaluator_to_search(behavior, root_result)
    with pytest.raises(SearchCapabilityError):
        compare_search_events(root_result)


def test_raw_evaluator_preference_compares_separately_with_reference_and_official_lc0():
    board = LczeroBoard()
    evaluator = evaluator_behavior(board, FixedEvaluator()(board))
    reference = ReferenceMCTS(c_puct=1.0).search(board, FixedEvaluator(), simulations=4)
    fixture = Path(__file__).parents[1] / "assets" / "lc0_root_snapshot_v1.txt"
    official = Lc0RootSnapshotParser().parse(
        fixture.read_text().splitlines(),
        request=Lc0SearchRequest(ROOT_FEN, nodes=256, options={"VerboseMoveStats": True}),
        engine_version="v0.31.2",
        network="fixture",
    )

    reference_comparison = compare_evaluator_to_search(evaluator, reference)
    official_comparison = compare_evaluator_to_search(evaluator, official)

    assert reference_comparison.source == "lczerolens-reference-mcts"
    assert reference_comparison.capability is SearchCapability.REPLAYABLE
    assert official_comparison.source == "official_lc0_uci"
    assert official_comparison.capability is SearchCapability.ROOT_SNAPSHOTS
    assert official_comparison.evaluator_probability_coverage < 1.0
    assert compare_search_events(reference, validate_replay=True).replay_validated
    with pytest.raises(SearchCapabilityError):
        compare_search_events(official)


def test_budgeted_snapshots_report_rank_q_selection_discovery_and_pv_evolution():
    board = LczeroBoard()
    evaluator = evaluator_behavior(board, output_for(board, {"e2e4": 3.0, "d2d4": 2.0}))

    def action(move, visits, q, pv):
        return RootAction(
            EdgeStatistics(move, ValuePerspective.ROOT_PLAYER, prior=0.5, visits=visits, mean_value=q),
            principal_variation=PrincipalVariation(pv),
        )

    trace = SearchTrace(
        ROOT_FEN,
        ChessPlayer.WHITE,
        SearchCapability.ROOT_SNAPSHOTS,
        SearchProvenance("synthetic-root-snapshots"),
        (
            RootSnapshot(
                0,
                RootSelection("e2e4", "fixture", "fixture"),
                budget=SearchBudget(SearchBudgetUnit.NODES, observed=0),
                actions=(action("e2e4", 0, 0.0, ("e2e4", "e7e5")), action("d2d4", 0, 0.0, ("d2d4", "d7d5"))),
            ),
            RootSnapshot(
                1,
                RootSelection("d2d4", "fixture", "fixture"),
                budget=SearchBudget(SearchBudgetUnit.NODES, observed=3),
                actions=(action("e2e4", 1, 0.2, ("e2e4", "c7c5")), action("d2d4", 2, 0.5, ("d2d4", "d7d5"))),
            ),
        ),
    )

    comparison = compare_evaluator_to_search(evaluator, trace)

    assert comparison.snapshots[0].ranks == (("e2e4", 1), ("d2d4", 1))
    assert comparison.snapshots[1].q_values == (("e2e4", 0.2), ("d2d4", 0.5))
    assert comparison.selected_move_changes == ((1, "e2e4", "d2d4"),)
    assert all(budget.observed == 3 for _, budget in comparison.discovery_budgets)
    assert dict(comparison.pv_stability) == {"d2d4": 1.0, "e2e4": 0.5}


def test_decision_comparison_links_both_candidates_to_variation_evidence():
    board = LczeroBoard()
    evaluator = evaluator_behavior(board, output_for(board, {"e2e4": 4.0, "d2d4": 2.0}))
    trace = ReferenceMCTS(c_puct=0.0).search(board, FixedEvaluator(), simulations=1)
    search_move = trace.snapshots[-1].selection.move
    evidence = {
        move: analyze_variation(
            board,
            (chess.Move.from_uci(move),),
            MaterialAnalyzer(FactPerspective.WHITE),
            MaterialAnalyzer(FactPerspective.BLACK),
        )
        for move in (evaluator.selected_move, search_move)
    }

    decision = compare_search_decision(evaluator, trace, variation_evidence=evidence)

    assert decision.evaluator_candidate == "e2e4"
    assert decision.search_candidate == search_move
    assert decision.evaluator_variation.moves[0].uci() == "e2e4"
    assert decision.search_variation.moves[0].uci() == search_move


def test_every_metric_definition_states_all_required_semantics():
    assert set(METRIC_DEFINITIONS) == set(BehaviorMetric)
    for definition in METRIC_DEFINITIONS.values():
        assert all(
            (
                definition.perspective,
                definition.normalization,
                definition.aggregation,
                definition.ties,
                definition.missing_or_illegal,
            )
        )
    assert METRIC_DEFINITIONS[BehaviorMetric.EVENT_PATH_DEPTH].required_capability is SearchCapability.FULL_EVENTS
    assert METRIC_DEFINITIONS[BehaviorMetric.REPLAY_VALIDATION].required_capability is SearchCapability.REPLAYABLE
