"""Tests for exact move and line analysis."""

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
from lczerolens.moves import (
    EvidenceTransition,
    EvidenceTransitionKind,
    ExactMoveEffect,
    HistoryPolicy,
    LineAnalysisError,
    LineFailureReason,
    LineIntent,
    LineRole,
    LineTerminal,
    analyze_move,
    analyze_line,
    exact_move_analyzers,
)


def test_capture_changes_material_and_retains_move_and_position_provenance():
    board = chess.Board("4k3/p7/8/8/8/8/8/R3K3 w - - 0 1")
    move = chess.Move.from_uci("a1a7")
    delta = analyze_move(
        board,
        "a1a7",
        MaterialAnalyzer(FactPerspective.WHITE),
        MaterialAnalyzer(FactPerspective.BLACK),
    )

    assert delta.before.fen == board.fen()
    assert delta.after.turn.value == "black"
    assert delta.move == move
    assert delta.facts_before is delta.before.evidence
    assert delta.facts_after is delta.after.evidence
    assert delta.evidence.captured_piece == chess.Piece(chess.PAWN, chess.BLACK)
    assert delta.evidence.capture_square == chess.A7
    assert delta.effects == (ExactMoveEffect.CAPTURE,)
    assert delta.removed.items[0].value == 1
    assert delta.created.items[0].value == 0
    assert delta.preserved.items[0].value == 5
    assert delta.changed[0].kind is EvidenceTransitionKind.CHANGED


def test_check_and_evasion_are_exact_move_effects():
    checking = chess.Board("4k3/8/8/8/8/8/R7/4K3 w - - 0 1")
    check = analyze_move(checking, chess.Move.from_uci("a2e2"), CheckStatusAnalyzer(FactPerspective.BLACK))
    assert check.effects == (ExactMoveEffect.CHECK,)
    assert check.created.items[0].value is True

    evading = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    evasion = analyze_move(evading, chess.Move.from_uci("e8f8"), CheckStatusAnalyzer(FactPerspective.BLACK))
    assert evasion.effects == (ExactMoveEffect.EVASION,)
    assert evasion.created.items[0].value is False


def test_discovered_attack_and_defender_removal_change_supporting_evidence():
    discovered = chess.Board("r3k3/8/8/8/8/8/B7/R3K3 w - - 0 1")
    attack = analyze_move(
        discovered,
        chess.Move.from_uci("a2b3"),
        AttacksDefendersAnalyzer(chess.A8, FactPerspective.BLACK),
    )
    assert attack.removed.items[0].value == AttacksDefendersValue(0, 0)
    assert attack.created.items[0].value == AttacksDefendersValue(1, 0)
    assert chess.A1 in attack.created.items[0].supporting_squares

    defended = chess.Board("4k3/5n2/8/4P3/2N5/8/8/4K3 w - - 0 1")
    defender = analyze_move(
        defended,
        chess.Move.from_uci("c4b6"),
        AttacksDefendersAnalyzer(chess.E5, FactPerspective.WHITE),
    )
    assert defender.removed.items[0].value == AttacksDefendersValue(1, 1)
    assert defender.created.items[0].value == AttacksDefendersValue(1, 0)


def test_equal_attack_counts_still_report_changed_attacker_identity():
    board = chess.Board("r3k3/8/8/8/8/8/R7/R3K3 w - - 0 1")
    delta = analyze_move(
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
    delta = analyze_move(chess.Board(fen), chess.Move.from_uci(uci), MaterialAnalyzer())

    assert delta.effects == effects
    assert delta.evidence.capture_square == capture_square
    assert (delta.evidence.rook_move.uci() if delta.evidence.rook_move else None) == rook_uci


def test_chess960_castling_moves_the_configured_rook_without_a_capture():
    board = chess.Board(None, chess960=True)
    board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.castling_rights = chess.BB_H1
    move = chess.Move(chess.G1, chess.H1)

    delta = analyze_move(board, move, MaterialAnalyzer())

    assert delta.effects == (ExactMoveEffect.CASTLING,)
    assert delta.evidence.captured_piece is None
    assert delta.evidence.capture_square is None
    assert delta.evidence.rook_move == chess.Move(chess.H1, chess.F1)


def test_unchanged_control_is_preserved_without_losing_original_evidence():
    board = chess.Board()
    before = MaterialAnalyzer(FactPerspective.WHITE).analyze(board)
    delta = analyze_move(board, chess.Move.from_uci("e2e4"), MaterialAnalyzer(FactPerspective.WHITE))

    assert not delta.created.items
    assert not delta.removed.items
    assert delta.preserved.items == (before,)
    assert delta.transitions[0].before == before
    assert delta.transitions[0].after.value == before.value
    assert delta.transitions[0].after.supporting_pieces != before.supporting_pieces
    assert delta.transitions[0].kind is EvidenceTransitionKind.PRESERVED


def test_default_suite_covers_both_sides_all_squares_and_mobility():
    analyzers = exact_move_analyzers()
    delta = analyze_move(chess.Board(), chess.Move.from_uci("e2e4"))

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
    with pytest.raises(TypeError, match="Move analysis"):
        analyze_move("not a board", chess.Move.from_uci("e2e4"))
    with pytest.raises(LineAnalysisError) as illegal:
        analyze_move(chess.Board(), "not a move")
    assert illegal.value.reason is LineFailureReason.ILLEGAL_MOVE
    assert illegal.value.move is None
    malformed = chess.Move(64, chess.E4)
    with pytest.raises(LineAnalysisError) as malformed_delta:
        analyze_move(chess.Board(), malformed)
    assert malformed_delta.value.reason is LineFailureReason.ILLEGAL_MOVE
    assert malformed_delta.value.move == malformed
    with pytest.raises(LineAnalysisError) as malformed_variation:
        analyze_line(chess.Board(), (malformed,))
    assert malformed_variation.value.reason is LineFailureReason.ILLEGAL_MOVE
    assert malformed_variation.value.ply_index == 0
    with pytest.raises(TypeError, match="Line analysis"):
        analyze_line("not a board", (chess.Move.from_uci("e2e4"),))
    with pytest.raises(ValueError, match="history_policy"):
        analyze_line(chess.Board(), (chess.Move.from_uci("e2e4"),), history_policy="allow")


def test_variation_tracks_alternating_perspective_terminal_result_and_intent():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    candidate = chess.Move.from_uci("f7g7")
    intent = LineIntent(
        role=LineRole.CANDIDATE_SUPPORT,
        claim_id="mate-candidate",
        candidate=candidate,
    )
    variation = analyze_line(board, (candidate,), CheckStatusAnalyzer(), intent=intent)

    assert variation.initial_position.turn.value == "white"
    assert variation.final_position.turn.value == "black"
    assert variation.initial_position.history_truncated
    assert variation.moves == (candidate,)
    assert variation.intent == intent
    assert variation.terminal.is_terminal
    assert variation.terminal.result == "1-0"
    assert variation.terminal.winner.value == "white"
    assert variation.terminal.termination is chess.Termination.CHECKMATE
    assert not variation.terminal.claimable_draw


def test_claimable_draw_is_not_reported_as_a_terminal_result():
    moves = tuple(chess.Move.from_uci(uci) for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2)

    variation = analyze_line(chess.Board(), moves, MaterialAnalyzer())

    assert variation.final_position.history_complete
    assert not variation.terminal.is_terminal
    assert variation.terminal.result == "*"
    assert variation.terminal.termination is None
    assert variation.terminal.claimable_draw


def test_refutation_intent_names_candidate_and_opponent_response():
    candidate = chess.Move.from_uci("e2e4")
    response = chess.Move.from_uci("e7e5")
    intent = LineIntent(
        role=LineRole.OPPONENT_REFUTATION,
        claim_id="candidate-keeps-e5-empty",
        candidate=candidate,
        response=response,
    )
    variation = analyze_line(chess.Board(), ("e2e4", "e7e5"), MaterialAnalyzer(), intent=intent)

    assert variation.intent.response == response
    assert [step.evidence.mover.value for step in variation.steps] == ["white", "black"]
    assert variation.final_position.ply == 2


def test_illegal_history_incompatible_and_intent_mismatch_fail_structurally():
    with pytest.raises(LineAnalysisError) as illegal:
        analyze_line(chess.Board(), (chess.Move.from_uci("e2e5"),), MaterialAnalyzer())
    assert illegal.value.reason is LineFailureReason.ILLEGAL_MOVE
    assert illegal.value.ply_index == 0
    assert illegal.value.move == chess.Move.from_uci("e2e5")
    assert illegal.value.fen == chess.Board().fen()

    with pytest.raises(LineAnalysisError, match="not-a-move") as malformed_string:
        analyze_line(chess.Board(), ("not-a-move",), MaterialAnalyzer())
    assert malformed_string.value.move is None

    truncated = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
    with pytest.raises(LineAnalysisError) as history:
        analyze_line(
            truncated,
            (chess.Move.from_uci("e2e3"),),
            MaterialAnalyzer(),
            history_policy=HistoryPolicy.REQUIRE_COMPLETE,
        )
    assert history.value.reason is LineFailureReason.HISTORY_INCOMPATIBLE

    intent = LineIntent(
        role=LineRole.CANDIDATE_SUPPORT,
        claim_id="different-candidate",
        candidate=chess.Move.from_uci("d2d4"),
    )
    with pytest.raises(LineAnalysisError) as mismatch:
        analyze_line(chess.Board(), (chess.Move.from_uci("e2e4"),), MaterialAnalyzer(), intent=intent)
    assert mismatch.value.reason is LineFailureReason.INTENT_MISMATCH


def test_empty_line_and_malformed_intents_are_rejected():
    with pytest.raises(LineAnalysisError) as empty:
        analyze_line(chess.Board(), (), MaterialAnalyzer())
    assert empty.value.reason is LineFailureReason.EMPTY_LINE
    with pytest.raises(ValueError, match="cannot name"):
        LineIntent(claim_id="claim")
    with pytest.raises(ValueError, match="LineRole"):
        LineIntent(role="neutral")
    with pytest.raises(ValueError, match="claim_id and candidate"):
        LineIntent(role=LineRole.CANDIDATE_SUPPORT)
    with pytest.raises(ValueError, match="does not take"):
        LineIntent(
            role=LineRole.CANDIDATE_SUPPORT,
            claim_id="claim",
            candidate=chess.Move.from_uci("e2e4"),
            response=chess.Move.from_uci("e7e5"),
        )
    with pytest.raises(ValueError, match="needs the opponent response"):
        LineIntent(
            role=LineRole.OPPONENT_REFUTATION,
            claim_id="claim",
            candidate=chess.Move.from_uci("e2e4"),
        )


def test_move_and_line_records_reject_incoherent_structure():
    move = analyze_move(chess.Board(), "e2e4", MaterialAnalyzer())
    with pytest.raises(ValueError, match="mover"):
        replace(move, evidence=replace(move.evidence, mover=move.after.turn))
    with pytest.raises(ValueError, match="one ply"):
        replace(move, after=replace(move.after, ply=move.before.ply))

    line = analyze_line(chess.Board(), ("e2e4", "e7e5"), MaterialAnalyzer())
    alternative = analyze_line(chess.Board(), ("d2d4",), MaterialAnalyzer())
    with pytest.raises(ValueError, match="at least one"):
        replace(line, steps=())
    with pytest.raises(ValueError, match="endpoints"):
        replace(line, final_position=line.initial_position)
    with pytest.raises(ValueError, match="continuous"):
        replace(
            line,
            final_position=alternative.final_position,
            steps=(line.steps[0], alternative.steps[0]),
        )

    with pytest.raises(ValueError, match="termination"):
        LineTerminal(True, "*", None, chess.Termination.CHECKMATE, False)
    with pytest.raises(ValueError, match="Non-terminal"):
        LineTerminal(False, "1-0", None, None, False)
