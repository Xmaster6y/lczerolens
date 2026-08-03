"""Tests for exact move deltas and variation evidence."""

from dataclasses import replace

import chess
import pytest

from lczerolens.facts import (
    AttacksDefendersAnalyzer,
    AttacksDefendersValue,
    CheckStatusAnalyzer,
    FactPerspective,
    MaterialAnalyzer,
)
from lczerolens.move_evidence import (
    EvidenceTransition,
    EvidenceTransitionKind,
    ExactMoveEffect,
    HistoryPolicy,
    VariationAnalysisError,
    VariationFailureReason,
    VariationIntent,
    VariationRole,
    analyze_move_delta,
    analyze_variation,
    exact_move_analyzers,
)


def test_capture_changes_material_and_retains_move_and_position_provenance():
    board = chess.Board("4k3/p7/8/8/8/8/8/R3K3 w - - 0 1")
    move = chess.Move.from_uci("a1a7")
    delta = analyze_move_delta(
        board,
        move,
        MaterialAnalyzer(FactPerspective.WHITE),
        MaterialAnalyzer(FactPerspective.BLACK),
    )

    assert delta.before.fen == board.fen()
    assert delta.after.turn.value == "black"
    assert delta.move.move == move
    assert delta.move.captured_piece == chess.Piece(chess.PAWN, chess.BLACK)
    assert delta.move.capture_square == chess.A7
    assert delta.move.effects == (ExactMoveEffect.CAPTURE,)
    assert delta.removed.items[0].value == 1
    assert delta.created.items[0].value == 0
    assert delta.preserved.items[0].value == 5
    assert delta.changed[0].kind is EvidenceTransitionKind.CHANGED


def test_check_and_evasion_are_exact_move_effects():
    checking = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    check = analyze_move_delta(checking, chess.Move.from_uci("a2e2"), CheckStatusAnalyzer(FactPerspective.BLACK))
    assert check.move.effects == (ExactMoveEffect.CHECK,)
    assert check.created.items[0].value is True

    evading = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    evasion = analyze_move_delta(evading, chess.Move.from_uci("e8f8"), CheckStatusAnalyzer(FactPerspective.BLACK))
    assert evasion.move.effects == (ExactMoveEffect.EVASION,)
    assert evasion.created.items[0].value is False


def test_discovered_attack_and_defender_removal_change_supporting_evidence():
    discovered = chess.Board("r3k3/8/8/8/8/8/B7/R3K3 w - - 0 1")
    attack = analyze_move_delta(
        discovered,
        chess.Move.from_uci("a2b3"),
        AttacksDefendersAnalyzer(chess.A8, FactPerspective.BLACK),
    )
    assert attack.removed.items[0].value == AttacksDefendersValue(0, 0)
    assert attack.created.items[0].value == AttacksDefendersValue(1, 0)
    assert chess.A1 in attack.created.items[0].supporting_squares

    defended = chess.Board("4k3/5n2/8/4P3/2N5/8/8/4K3 w - - 0 1")
    defender = analyze_move_delta(
        defended,
        chess.Move.from_uci("c4b6"),
        AttacksDefendersAnalyzer(chess.E5, FactPerspective.WHITE),
    )
    assert defender.removed.items[0].value == AttacksDefendersValue(1, 1)
    assert defender.created.items[0].value == AttacksDefendersValue(1, 0)


def test_equal_attack_counts_still_report_changed_attacker_identity():
    board = chess.Board("r3k3/8/8/8/8/8/R7/R3K3 w - - 0 1")
    delta = analyze_move_delta(
        board,
        chess.Move.from_uci("a2b2"),
        AttacksDefendersAnalyzer(chess.A8, FactPerspective.BLACK),
    )

    assert delta.removed.items[0].value == delta.created.items[0].value == AttacksDefendersValue(1, 0)
    assert delta.removed.items[0].supporting_pieces[0].square == chess.A2
    assert delta.created.items[0].supporting_pieces[0].square == chess.A1
    assert delta.changed[0].kind is EvidenceTransitionKind.CHANGED


@pytest.mark.parametrize(
    ("fen", "uci", "effects", "capture_square", "rook_uci"),
    [
        (
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "a7a8q",
            (ExactMoveEffect.CHECK, ExactMoveEffect.PROMOTION),
            None,
            None,
        ),
        (
            "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
            "e1g1",
            (ExactMoveEffect.CASTLING,),
            None,
            "h1f1",
        ),
        (
            "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
            "e1c1",
            (ExactMoveEffect.CASTLING,),
            None,
            "a1d1",
        ),
        (
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
            "e5d6",
            (ExactMoveEffect.CAPTURE, ExactMoveEffect.EN_PASSANT),
            chess.D5,
            None,
        ),
        (
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
            "e4d3",
            (ExactMoveEffect.CAPTURE, ExactMoveEffect.EN_PASSANT),
            chess.D4,
            None,
        ),
    ],
)
def test_special_moves_have_explicit_exact_metadata(fen, uci, effects, capture_square, rook_uci):
    delta = analyze_move_delta(chess.Board(fen), chess.Move.from_uci(uci), MaterialAnalyzer())

    assert delta.move.effects == effects
    assert delta.move.capture_square == capture_square
    assert (delta.move.rook_move.uci() if delta.move.rook_move else None) == rook_uci


def test_unchanged_control_is_preserved_without_losing_original_evidence():
    board = chess.Board()
    before = MaterialAnalyzer(FactPerspective.WHITE).analyze(board)
    delta = analyze_move_delta(board, chess.Move.from_uci("e2e4"), MaterialAnalyzer(FactPerspective.WHITE))

    assert not delta.created.items
    assert not delta.removed.items
    assert delta.preserved.items == (before,)
    assert delta.transitions[0].before == before
    assert delta.transitions[0].after.value == before.value
    assert delta.transitions[0].after.supporting_pieces != before.supporting_pieces
    assert delta.transitions[0].kind is EvidenceTransitionKind.PRESERVED


def test_default_suite_covers_both_sides_all_squares_and_mobility():
    analyzers = exact_move_analyzers()
    delta = analyze_move_delta(chess.Board(), chess.Move.from_uci("e2e4"))

    assert len(analyzers) == 145
    assert len(delta.before.evidence) == len(delta.after.evidence) == 145
    assert delta.before.turn.value == "white"
    assert delta.after.turn.value == "black"
    assert delta.before.history_complete
    assert delta.after.history_complete


def test_transition_and_input_validation_rejects_inconsistent_records():
    evidence = MaterialAnalyzer(FactPerspective.WHITE).analyze(chess.Board())
    changed = replace(evidence, value=999)
    with pytest.raises(ValueError, match="same fact identity and value"):
        EvidenceTransition(evidence, changed, EvidenceTransitionKind.PRESERVED)
    with pytest.raises(ValueError, match="different fact identity or value"):
        EvidenceTransition(evidence, evidence, EvidenceTransitionKind.CHANGED)
    with pytest.raises(TypeError, match="Move-delta analysis"):
        analyze_move_delta("not a board", chess.Move.from_uci("e2e4"))
    with pytest.raises(VariationAnalysisError) as illegal:
        analyze_move_delta(chess.Board(), "not a move")
    assert illegal.value.reason is VariationFailureReason.ILLEGAL_MOVE
    assert illegal.value.move is None
    with pytest.raises(TypeError, match="Variation analysis"):
        analyze_variation("not a board", (chess.Move.from_uci("e2e4"),))
    with pytest.raises(ValueError, match="history_policy"):
        analyze_variation(chess.Board(), (chess.Move.from_uci("e2e4"),), history_policy="allow")


def test_variation_tracks_alternating_perspective_terminal_result_and_intent():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    candidate = chess.Move.from_uci("f7g7")
    intent = VariationIntent(
        role=VariationRole.CANDIDATE_SUPPORT,
        claim_id="mate-candidate",
        candidate=candidate,
    )
    variation = analyze_variation(board, (candidate,), CheckStatusAnalyzer(), intent=intent)

    assert variation.initial.turn.value == "white"
    assert variation.final.turn.value == "black"
    assert variation.initial.history_truncated
    assert variation.moves == (candidate,)
    assert variation.intent == intent
    assert variation.terminal.is_terminal
    assert variation.terminal.result == "1-0"
    assert variation.terminal.winner.value == "white"
    assert variation.terminal.termination is chess.Termination.CHECKMATE


def test_refutation_intent_names_candidate_and_opponent_response():
    candidate = chess.Move.from_uci("e2e4")
    response = chess.Move.from_uci("e7e5")
    intent = VariationIntent(
        role=VariationRole.OPPONENT_REFUTATION,
        claim_id="candidate-keeps-e5-empty",
        candidate=candidate,
        response=response,
    )
    variation = analyze_variation(chess.Board(), (candidate, response), MaterialAnalyzer(), intent=intent)

    assert variation.intent.response == response
    assert [delta.move.mover.value for delta in variation.deltas] == ["white", "black"]
    assert variation.final.ply == 2


def test_illegal_history_incompatible_and_intent_mismatch_fail_structurally():
    with pytest.raises(VariationAnalysisError) as illegal:
        analyze_variation(chess.Board(), (chess.Move.from_uci("e2e5"),), MaterialAnalyzer())
    assert illegal.value.reason is VariationFailureReason.ILLEGAL_MOVE
    assert illegal.value.ply_index == 0
    assert illegal.value.move == chess.Move.from_uci("e2e5")
    assert illegal.value.fen == chess.Board().fen()

    truncated = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    with pytest.raises(VariationAnalysisError) as history:
        analyze_variation(
            truncated,
            (chess.Move.from_uci("e2e3"),),
            MaterialAnalyzer(),
            history_policy=HistoryPolicy.REQUIRE_COMPLETE,
        )
    assert history.value.reason is VariationFailureReason.HISTORY_INCOMPATIBLE

    intent = VariationIntent(
        role=VariationRole.CANDIDATE_SUPPORT,
        claim_id="different-candidate",
        candidate=chess.Move.from_uci("d2d4"),
    )
    with pytest.raises(VariationAnalysisError) as mismatch:
        analyze_variation(chess.Board(), (chess.Move.from_uci("e2e4"),), MaterialAnalyzer(), intent=intent)
    assert mismatch.value.reason is VariationFailureReason.INTENT_MISMATCH


def test_empty_line_and_malformed_intents_are_rejected():
    with pytest.raises(VariationAnalysisError) as empty:
        analyze_variation(chess.Board(), (), MaterialAnalyzer())
    assert empty.value.reason is VariationFailureReason.EMPTY_LINE
    with pytest.raises(ValueError, match="cannot name"):
        VariationIntent(claim_id="claim")
    with pytest.raises(ValueError, match="VariationRole"):
        VariationIntent(role="neutral")
    with pytest.raises(ValueError, match="claim_id and candidate"):
        VariationIntent(role=VariationRole.CANDIDATE_SUPPORT)
    with pytest.raises(ValueError, match="does not take"):
        VariationIntent(
            role=VariationRole.CANDIDATE_SUPPORT,
            claim_id="claim",
            candidate=chess.Move.from_uci("e2e4"),
            response=chess.Move.from_uci("e7e5"),
        )
    with pytest.raises(ValueError, match="needs the opponent response"):
        VariationIntent(
            role=VariationRole.OPPONENT_REFUTATION,
            claim_id="claim",
            candidate=chess.Move.from_uci("e2e4"),
        )
