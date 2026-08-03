"""Tests for evidence-bearing chess facts and exact reference analyzers."""

from dataclasses import replace

import chess
import pytest

from lczerolens.facts import (
    AnalyzerProvenance,
    AttacksDefendersAnalyzer,
    AttacksDefendersValue,
    CheckStatusAnalyzer,
    Evidence,
    EvidenceSet,
    FactAnalyzer,
    FactKind,
    FactPerspective,
    FactScope,
    Guarantee,
    GuaranteeMismatchError,
    HistoryRequirement,
    LegalMobilityAnalyzer,
    MaterialAnalyzer,
    PiecePresenceAnalyzer,
    SideSubject,
    UndefinedReason,
    analyze_facts,
    history_is_available,
)


def test_reference_analyzers_return_uniform_exact_evidence():
    board = chess.Board("4k3/5n2/8/4P3/2N5/8/8/4K3 w - - 0 1")
    evidence = analyze_facts(
        board,
        MaterialAnalyzer(),
        PiecePresenceAnalyzer(chess.KNIGHT),
        AttacksDefendersAnalyzer(chess.E5),
        CheckStatusAnalyzer(),
        LegalMobilityAnalyzer(),
    )

    assert len(evidence) == 5
    assert all(item.guarantee is Guarantee.EXACT for item in evidence)
    assert all(item.perspective is FactPerspective.SIDE_TO_MOVE for item in evidence)
    assert all(item.provenance.version == "1" for item in evidence)
    assert evidence.items[0].value == 4
    assert evidence.items[1].value is True
    assert evidence.items[2].value == AttacksDefendersValue(1, 1)
    assert evidence.items[2].supporting_squares == (chess.E5, chess.F7, chess.C4)
    assert evidence.items[3].value is False
    assert evidence.items[4].value == len(evidence.items[4].supporting_moves)
    assert all(isinstance(analyzer, FactAnalyzer) for analyzer in (MaterialAnalyzer(), LegalMobilityAnalyzer()))


def test_material_and_piece_presence_are_symmetric_by_absolute_side():
    board = chess.Board("4k3/8/8/8/8/8/PP6/4K3 w - - 0 1")
    mirrored = board.mirror()

    white_material = MaterialAnalyzer(FactPerspective.WHITE).analyze(board)
    black_material = MaterialAnalyzer(FactPerspective.BLACK).analyze(mirrored)
    white_pawns = PiecePresenceAnalyzer(chess.PAWN, FactPerspective.WHITE).analyze(board)
    black_pawns = PiecePresenceAnalyzer(chess.PAWN, FactPerspective.BLACK).analyze(mirrored)

    assert white_material.value == black_material.value == 2
    assert len(white_material.supporting_pieces) == len(black_material.supporting_pieces) == 3
    assert white_pawns.value is black_pawns.value is True


def test_attack_check_and_mobility_analyzers_preserve_color_flip_symmetry():
    board = chess.Board("4k3/5n2/8/4P3/2N5/8/4R3/4K3 b - - 0 1")
    mirrored = board.mirror()

    attack = AttacksDefendersAnalyzer(chess.E5, FactPerspective.WHITE).analyze(board)
    mirrored_attack = AttacksDefendersAnalyzer(chess.E4, FactPerspective.BLACK).analyze(mirrored)
    check = CheckStatusAnalyzer(FactPerspective.BLACK).analyze(board)
    mirrored_check = CheckStatusAnalyzer(FactPerspective.WHITE).analyze(mirrored)
    mobility = LegalMobilityAnalyzer().analyze(board)
    mirrored_mobility = LegalMobilityAnalyzer().analyze(mirrored)

    assert attack.value == mirrored_attack.value
    assert check.value == mirrored_check.value
    assert mobility.value == mirrored_mobility.value


def test_check_evidence_identifies_king_and_checking_piece():
    board = chess.Board("4k3/8/8/8/8/8/4R3/4K3 b - - 0 1")
    evidence = CheckStatusAnalyzer().analyze(board)

    assert evidence.value is True
    assert evidence.supporting_squares == (chess.E8, chess.E2)
    assert evidence.supporting_pieces[0].piece == chess.Piece(chess.ROOK, chess.WHITE)
    assert evidence.supporting_pieces[0].role == "checking"


def test_legal_mobility_retains_each_legal_move():
    evidence = LegalMobilityAnalyzer().analyze(chess.Board())

    assert evidence.value == 20
    assert len(evidence.supporting_moves) == 20
    assert chess.Move.from_uci("e2e4") in evidence.supporting_moves


def test_malformed_position_is_explicitly_undefined_where_semantics_require_it():
    missing_king = chess.Board("8/8/8/8/8/8/8/4K3 b - - 0 1")

    check = CheckStatusAnalyzer().analyze(missing_king)
    mobility = LegalMobilityAnalyzer().analyze(missing_king)

    assert check.value is None
    assert check.undefined_reason is UndefinedReason.MISSING_KING
    assert mobility.value is None
    assert mobility.undefined_reason is UndefinedReason.INVALID_POSITION
    assert not check.is_defined


def test_history_loss_is_detectable_and_must_be_encoded_as_undefined():
    complete = chess.Board()
    complete.push_uci("e2e4")
    reconstructed = chess.Board(complete.fen())

    assert history_is_available(complete, HistoryRequirement.LAST_MOVE)
    assert history_is_available(complete, HistoryRequirement.FULL_MOVE_STACK)
    assert not history_is_available(reconstructed, HistoryRequirement.LAST_MOVE)
    assert not history_is_available(reconstructed, HistoryRequirement.FULL_MOVE_STACK)

    history_fact = Evidence(
        kind=FactKind.CHECK_STATUS,
        scope=FactScope.SIDE,
        subject=SideSubject(side=CheckStatusAnalyzer().analyze(reconstructed).subject.side),
        value=None,
        perspective=FactPerspective.SIDE_TO_MOVE,
        guarantee=Guarantee.EXACT,
        provenance=AnalyzerProvenance("test.last_move", "1"),
        history_requirement=HistoryRequirement.LAST_MOVE,
        history_available=False,
        undefined_reason=UndefinedReason.HISTORY_UNAVAILABLE,
    )
    assert not history_fact.is_defined


def test_evidence_invariants_reject_ambiguous_undefined_state():
    base = CheckStatusAnalyzer().analyze(chess.Board())

    with pytest.raises(ValueError, match="exactly one"):
        replace(base, value=None, undefined_reason=None)
    with pytest.raises(ValueError, match="Unavailable required history"):
        replace(
            base,
            history_requirement=HistoryRequirement.LAST_MOVE,
            history_available=False,
            undefined_reason=UndefinedReason.INVALID_POSITION,
            value=None,
        )


def test_composition_and_filtering_preserve_guarantees():
    exact = MaterialAnalyzer().analyze(chess.Board())
    heuristic = replace(
        exact,
        guarantee=Guarantee.HEURISTIC,
        provenance=AnalyzerProvenance("example.heuristic", "1"),
    )
    evidence = EvidenceSet((exact,)).compose(heuristic)

    assert evidence.filter(guarantee=Guarantee.EXACT).items == (exact,)
    assert evidence.filter(guarantee=Guarantee.HEURISTIC).items == (heuristic,)
    with pytest.raises(GuaranteeMismatchError, match="other than exact"):
        evidence.values(guarantee=Guarantee.EXACT)


def test_analyzers_reject_malformed_inputs_and_configuration():
    with pytest.raises(TypeError, match="python-chess Board"):
        MaterialAnalyzer().analyze("not a board")
    with pytest.raises(ValueError, match="piece type"):
        PiecePresenceAnalyzer(99)
    with pytest.raises(ValueError, match="square"):
        AttacksDefendersAnalyzer(99)
    with pytest.raises(ValueError, match="needs white, black"):
        MaterialAnalyzer(FactPerspective.ABSOLUTE).analyze(chess.Board())
