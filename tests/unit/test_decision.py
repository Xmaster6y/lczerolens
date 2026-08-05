"""Tests for the concrete decision and counterfactual comparison interfaces."""

from dataclasses import replace

import chess
import pytest
import torch
from torch import nn

from lczerolens import LczeroModel
from lczerolens._codec import encode_move
from lczerolens.counterfactuals import sibling_counterfactual
from lczerolens.decision import DecisionActions, compare_counterfactual, compare_decision
from lczerolens.evaluator import LczeroEvaluator
from lczerolens.moves import analyze_line
from lczerolens.provenance import ChessPlayer, EvaluationProvenance
from lczerolens.search_trace import (
    EdgeStatistics,
    PositionEvaluation,
    PrincipalVariation,
    RootAction,
    RootSelection,
    RootSnapshot,
    SearchCapability,
    SearchProvenance,
    SearchTrace,
    ValuePerspective,
)


class FixtureNetwork(nn.Module):
    def forward(self, planes):
        batch = planes.shape[0]
        policy = torch.zeros((batch, 1858), device=planes.device)
        board = chess.Board()
        policy[:, encode_move(board, chess.Move.from_uci("e2e4"))] = 4.0
        policy[:, encode_move(board, chess.Move.from_uci("d2d4"))] = 2.0
        return policy, torch.full((batch,), 0.25, device=planes.device)


def fixture_evaluator():
    return LczeroEvaluator(LczeroModel(FixtureNetwork(), out_keys=["policy", "value"]))


def fixture_trace(*, with_actions=True, selected="d2d4"):
    board = chess.Board()
    actions = None
    capability = SearchCapability.ROOT_RESULT
    if with_actions:
        capability = SearchCapability.ROOT_ACTION_STATS
        actions = (
            RootAction(
                EdgeStatistics(
                    "d2d4", ValuePerspective.ROOT_PLAYER, prior=0.4, visits=7, total_value=2.8, mean_value=0.4
                ),
                principal_variation=PrincipalVariation(("d2d4", "d7d5")),
            ),
            RootAction(
                EdgeStatistics(
                    "e2e4", ValuePerspective.ROOT_PLAYER, prior=0.6, visits=3, total_value=0.6, mean_value=0.2
                ),
                principal_variation=PrincipalVariation(("e2e4", "e7e5")),
            ),
        )
    return SearchTrace(
        root_fen=board.fen(),
        root_player=ChessPlayer.WHITE,
        capability=capability,
        provenance=SearchProvenance("fixture-search"),
        snapshots=(RootSnapshot(0, RootSelection(selected, "visits", "uci"), actions=actions),),
    )


def test_compare_decision_uses_evaluation_records_and_keeps_search_evidence_separate():
    evaluation = fixture_evaluator().evaluate(chess.Board())
    trace = fixture_trace()

    decision = compare_decision(evaluation, trace)

    assert decision.policy_move == "e2e4"
    assert decision.search_move == "d2d4"
    assert decision.changed
    assert decision.search is trace
    assert decision.evaluation == evaluation.record()
    assert decision.search_source == "fixture-search"
    assert decision.search_capability is SearchCapability.ROOT_ACTION_STATS
    assert decision.actions["e2e4"].policy_rank == 1
    assert decision.actions[chess.Move.from_uci("d2d4")].search_rank == 1
    assert decision.actions["d2d4"].search_visit_share == pytest.approx(0.7)
    assert decision.actions["d2d4"].line.moves == (
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("d7d5"),
    )
    assert decision.actions["e2e4"].selected_by_policy
    assert decision.actions["d2d4"].selected_by_search


def test_root_only_decision_preserves_unavailable_search_action_fields():
    record = fixture_evaluator().evaluate(chess.Board()).record()

    decision = compare_decision(record, fixture_trace(with_actions=False, selected="e2e4"))

    assert not decision.changed
    assert decision.actions["e2e4"].search_rank is None
    assert decision.actions["e2e4"].search_visits is None
    assert decision.actions["e2e4"].line.moves == (chess.Move.from_uci("e2e4"),)
    assert len(decision.actions) == 20


def test_supplied_line_is_retained_and_invalid_comparison_inputs_fail_closed():
    evaluation = fixture_evaluator().evaluate(chess.Board())
    line = analyze_line(chess.Board(), ("e2e4", "e7e5"))
    decision = compare_decision(evaluation, fixture_trace(), line_analyses={"e2e4": line})

    assert decision.actions["e2e4"].line is line
    wrong = chess.Board()
    wrong.halfmove_clock = 1
    with pytest.raises(ValueError, match="same root position"):
        compare_decision(evaluation, replace(fixture_trace(), root_fen=wrong.fen()))
    with pytest.raises(ValueError, match="legal evaluated root moves"):
        compare_decision(evaluation, fixture_trace(), line_analyses={"e2e5": line})
    with pytest.raises(ValueError, match="start at the root"):
        compare_decision(
            evaluation,
            fixture_trace(),
            line_analyses={"e2e4": analyze_line(chess.Board(), ("d2d4",))},
        )
    with pytest.raises(TypeError, match="Evaluation or EvaluationRecord"):
        compare_decision(object(), fixture_trace())
    with pytest.raises(TypeError, match="SearchTrace"):
        compare_decision(evaluation, object())


def test_compare_counterfactual_evaluates_reconstructable_factual_and_alternative_positions():
    pair = sibling_counterfactual(chess.Board(), factual="e2e4", alternative="d2d4")

    comparison = compare_counterfactual(pair, fixture_evaluator())

    assert comparison.pair is pair
    assert comparison.factual_evaluation.position.moves == ("e2e4",)
    assert comparison.alternative_evaluation.position.moves == ("d2d4",)
    assert comparison.policy_change.factual_move is not None
    assert comparison.policy_change.alternative_move is not None
    assert 0 <= comparison.policy_change.total_variation <= 1
    assert comparison.value_change.delta == pytest.approx(0.0)

    decision = compare_decision(
        fixture_evaluator().evaluate(chess.Board()),
        fixture_trace(),
        counterfactuals=(comparison,),
    )
    assert decision.counterfactuals == (comparison,)


def test_counterfactual_comparison_rejects_failed_pairs_and_invalid_evaluators():
    failed = sibling_counterfactual(chess.Board(), factual="e2e5")
    pair = sibling_counterfactual(chess.Board(), factual="e2e4", alternative="d2d4")

    with pytest.raises(ValueError, match="successfully constructed"):
        compare_counterfactual(failed, fixture_evaluator())
    with pytest.raises(TypeError, match="CounterfactualPair"):
        compare_counterfactual(object(), fixture_evaluator())
    with pytest.raises(TypeError, match="evaluation interface"):
        compare_counterfactual(pair, object())

    class WrongCollection:
        def evaluate(self, boards):
            return object()

    class WrongEntries:
        def evaluate(self, boards):
            return (object(), object())

    with pytest.raises(TypeError, match="exactly two"):
        compare_counterfactual(pair, WrongCollection())
    with pytest.raises(TypeError, match="Evaluation views"):
        compare_counterfactual(pair, WrongEntries())


def test_decision_action_collection_rejects_duplicates_and_noncanonical_order():
    decision = compare_decision(fixture_evaluator().evaluate(chess.Board()), fixture_trace())
    action = decision.actions["e2e4"]

    assert decision.actions == DecisionActions(tuple(decision.actions.values()))
    assert decision.actions != object()
    with pytest.raises(ValueError, match="unique"):
        DecisionActions((action, action))
    with pytest.raises(ValueError, match="canonical"):
        DecisionActions(tuple(reversed(tuple(decision.actions.values()))))
    comparison = compare_counterfactual(
        sibling_counterfactual(chess.Board(), factual="e2e4", alternative="d2d4"), fixture_evaluator()
    )
    assert comparison.policy_change[chess.Move.from_uci("e7e5")].move == "e7e5"


def test_decision_records_reject_inconsistent_state_and_missing_selections():
    evaluation = fixture_evaluator().evaluate(chess.Board())
    decision = compare_decision(evaluation, fixture_trace())
    wrong = chess.Board()
    wrong.halfmove_clock = 1

    with pytest.raises(ValueError, match="same root position"):
        replace(decision, search=replace(fixture_trace(), root_fen=wrong.fen()))
    with pytest.raises(ValueError, match="present in decision actions"):
        replace(decision, policy_move="a1a1")
    with pytest.raises(ValueError, match="changed status"):
        replace(decision, changed=not decision.changed)
    with pytest.raises(ValueError, match="CounterfactualComparison"):
        replace(decision, counterfactuals=(object(),))

    comparison = compare_counterfactual(
        sibling_counterfactual(chess.Board(), factual="e2e4", alternative="d2d4"), fixture_evaluator()
    )
    other_provenance = EvaluationProvenance("other", "fixture")
    mismatched = replace(
        comparison,
        factual_evaluation=replace(comparison.factual_evaluation, provenance=other_provenance),
    )
    with pytest.raises(ValueError, match="evaluator provenance"):
        replace(decision, counterfactuals=(mismatched,))

    root = chess.Board()
    no_selection = SearchTrace(
        root_fen=root.fen(),
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.ROOT_RESULT,
        provenance=SearchProvenance("fixture-search"),
        snapshots=(
            RootSnapshot(
                0,
                evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="exposed search selection"):
        compare_decision(evaluation, no_selection)

    terminal = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    terminal_evaluation = fixture_evaluator().evaluate(terminal)
    terminal_trace = SearchTrace(
        root_fen=terminal.fen(),
        root_player=ChessPlayer.BLACK,
        capability=SearchCapability.ROOT_RESULT,
        provenance=SearchProvenance("fixture-search"),
        snapshots=(
            RootSnapshot(
                0,
                evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="non-terminal evaluation policy"):
        compare_decision(terminal_evaluation, terminal_trace)
