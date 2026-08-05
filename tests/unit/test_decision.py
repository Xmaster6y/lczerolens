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
from lczerolens.search.result import SearchEvidenceUnavailable, SearchResult
from lczerolens.search.trace import (
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
        signal = planes[:, 6, 4, 3] - planes[:, 6, 4, 4]
        policy[:, encode_move(board, chess.Move.from_uci("e2e4"))] = 4.0 + signal
        policy[:, encode_move(board, chess.Move.from_uci("d2d4"))] = 2.0 - signal
        value = 0.5 * signal
        return policy, value


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


def fixture_search(*, with_actions=True, selected="d2d4"):
    return SearchResult.from_trace(fixture_trace(with_actions=with_actions, selected=selected))


def test_compare_decision_uses_evaluation_records_and_keeps_search_evidence_separate():
    evaluation = fixture_evaluator().evaluate(chess.Board())
    search = fixture_search()

    decision = compare_decision(evaluation, search)

    assert decision.policy_move == "e2e4"
    assert decision.search_move == "d2d4"
    assert decision.changed
    assert decision.search is search
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

    decision = compare_decision(record, fixture_search(with_actions=False, selected="e2e4"))

    assert not decision.changed
    assert decision.actions["e2e4"].search_rank is None
    assert decision.actions["e2e4"].search_visits is None
    assert decision.actions["e2e4"].line.moves == (chess.Move.from_uci("e2e4"),)
    assert len(decision.actions) == 20


def test_supplied_line_is_retained_and_invalid_comparison_inputs_fail_closed():
    evaluation = fixture_evaluator().evaluate(chess.Board())
    line = analyze_line(chess.Board(), ("e2e4", "e7e5"))
    decision = compare_decision(evaluation, fixture_search(), line_analyses={"e2e4": line})

    assert decision.actions["e2e4"].line is line
    wrong = chess.Board()
    wrong.halfmove_clock = 1
    with pytest.raises(ValueError, match="same root position"):
        compare_decision(evaluation, SearchResult.from_trace(replace(fixture_trace(), root_fen=wrong.fen())))
    with pytest.raises(ValueError, match="legal evaluated root moves"):
        compare_decision(evaluation, fixture_search(), line_analyses={"e2e5": line})
    with pytest.raises(ValueError, match="start at the root"):
        compare_decision(
            evaluation,
            fixture_search(),
            line_analyses={"e2e4": analyze_line(chess.Board(), ("d2d4",))},
        )
    with pytest.raises(TypeError, match="Evaluation or EvaluationRecord"):
        compare_decision(object(), fixture_search())
    with pytest.raises(TypeError, match="SearchResult"):
        compare_decision(evaluation, object())


def test_compare_counterfactual_evaluates_reconstructable_factual_and_alternative_positions():
    pair = sibling_counterfactual(chess.Board(), factual="e2e4", alternative="d2d4")

    comparison = compare_counterfactual(pair, fixture_evaluator())

    assert comparison.pair is pair
    assert comparison.factual_evaluation.position.moves == ("e2e4",)
    assert comparison.alternative_evaluation.position.moves == ("d2d4",)
    assert comparison.policy_change.factual_move is not None
    assert comparison.policy_change.alternative_move is not None
    assert 0 < comparison.policy_change.total_variation <= 1
    assert comparison.value_change.delta != pytest.approx(0.0)

    decision = compare_decision(
        fixture_evaluator().evaluate(chess.Board()),
        fixture_search(),
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
    decision = compare_decision(fixture_evaluator().evaluate(chess.Board()), fixture_search())
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
    decision = compare_decision(evaluation, fixture_search())
    wrong = chess.Board()
    wrong.halfmove_clock = 1

    with pytest.raises(ValueError, match="same root position"):
        replace(
            decision,
            search=SearchResult.from_trace(replace(fixture_trace(), root_fen=wrong.fen())),
        )
    with pytest.raises(ValueError, match="present in decision actions"):
        replace(decision, policy_move="a1a1")
    with pytest.raises(ValueError, match="changed status"):
        replace(decision, changed=not decision.changed)
    with pytest.raises(ValueError, match="CounterfactualComparison"):
        replace(decision, counterfactuals=(object(),))
    forged = replace(decision.actions["e2e4"], policy_probability=0.0)
    forged_actions = DecisionActions(
        tuple(forged if move == "e2e4" else action for move, action in decision.actions.items())
    )
    with pytest.raises(ValueError, match="must match their evaluator and search evidence"):
        replace(decision, actions=forged_actions)

    comparison = compare_counterfactual(
        sibling_counterfactual(chess.Board(), factual="e2e4", alternative="d2d4"), fixture_evaluator()
    )
    other_provenance = EvaluationProvenance("other", "fixture")
    with pytest.raises(ValueError, match="same evaluator provenance"):
        replace(
            comparison,
            factual_evaluation=replace(comparison.factual_evaluation, provenance=other_provenance),
        )

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
    with pytest.raises(SearchEvidenceUnavailable, match="selected move"):
        SearchResult.from_trace(no_selection)


def test_decision_and_counterfactual_records_reject_forged_direct_construction():
    decision = compare_decision(fixture_evaluator().evaluate(chess.Board()), fixture_search())
    comparison = compare_counterfactual(
        sibling_counterfactual(chess.Board(), factual="e2e4", alternative="d2d4"), fixture_evaluator()
    )

    with pytest.raises(ValueError, match="evaluation and search evidence"):
        replace(decision, evaluation=object())
    with pytest.raises(ValueError, match="DecisionActions"):
        replace(decision, actions=object())
    with pytest.raises(ValueError, match="exactly match"):
        replace(decision, actions=DecisionActions(tuple(decision.actions.values())[:-1]))

    unrelated_line = analyze_line(chess.Board(), ("d2d4",))
    forged_line_actions = DecisionActions(
        tuple(
            replace(action, line=unrelated_line) if move == "e2e4" else action
            for move, action in decision.actions.items()
        )
    )
    with pytest.raises(ValueError, match="start at the root"):
        replace(decision, actions=forged_line_actions)

    other = EvaluationProvenance("other", "fixture")
    other_comparison = replace(
        comparison,
        factual_evaluation=replace(comparison.factual_evaluation, provenance=other),
        alternative_evaluation=replace(comparison.alternative_evaluation, provenance=other),
    )
    with pytest.raises(ValueError, match="decision evaluator provenance"):
        replace(decision, counterfactuals=(other_comparison,))

    failed = sibling_counterfactual(chess.Board(), factual="e2e5")
    with pytest.raises(ValueError, match="successful pair"):
        replace(comparison, pair=failed)
    with pytest.raises(ValueError, match="evaluation records"):
        replace(comparison, factual_evaluation=object())
    with pytest.raises(ValueError, match="factual evaluation"):
        replace(comparison, factual_evaluation=comparison.alternative_evaluation)
    with pytest.raises(ValueError, match="alternative evaluation"):
        replace(comparison, alternative_evaluation=comparison.factual_evaluation)
    with pytest.raises(ValueError, match="policy change"):
        replace(
            comparison,
            policy_change=replace(comparison.policy_change, total_variation=0.0),
        )
    with pytest.raises(ValueError, match="value change"):
        replace(
            comparison,
            value_change=replace(comparison.value_change, delta=99.0),
        )
