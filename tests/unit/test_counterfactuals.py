"""Tests for constraint-aware chess counterfactuals."""

import chess
import pytest

import lczerolens.counterfactuals as counterfactuals

from lczerolens.counterfactuals import (
    ConstraintRelation,
    CounterfactualConstraints,
    CounterfactualFailureReason,
    CounterfactualValidity,
    PositionAttribute,
    RelocatePieceOperator,
    RemovePieceOperator,
    SiblingMoveOperator,
    relocate_piece_counterfactual,
    remove_piece_counterfactual,
    sibling_counterfactual,
)
from lczerolens.facts import AttacksDefendersAnalyzer, FactPerspective, MaterialAnalyzer


def test_sibling_moves_are_deterministic_history_consistent_and_evidence_bearing():
    parent = chess.Board()
    constraints = CounterfactualConstraints(
        changed_attributes=frozenset((PositionAttribute.EN_PASSANT,)),
        preserved_attributes=frozenset(
            (PositionAttribute.TURN, PositionAttribute.KINGS, PositionAttribute.MATERIAL, PositionAttribute.HISTORY)
        ),
        preserved_facts=(MaterialAnalyzer(FactPerspective.WHITE),),
        changed_facts=(AttacksDefendersAnalyzer(chess.E3, FactPerspective.WHITE),),
    )

    result = sibling_counterfactual(
        parent,
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("d2d4"),
        constraints=constraints,
    )

    assert result.succeeded
    assert result.validity is CounterfactualValidity.HISTORY_CONSISTENT
    assert result.history.reachability_proven
    assert result.history.shared_parent
    assert result.history.legal_from_shared_parent
    assert result.modified is not None
    assert result.modified.rule_valid
    assert result.modified.history_complete
    assert result.original.fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert result.modified.fen == "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1"
    assert result.changed_attributes[0].satisfied
    assert all(item.satisfied for item in result.preserved_attributes)
    assert result.preserved_facts[0].original.value == result.preserved_facts[0].modified.value == 39
    assert result.changed_facts[0].original.value == result.changed_facts[0].modified.value
    assert result.changed_facts[0].original.supporting_pieces != result.changed_facts[0].modified.supporting_pieces
    assert result.original_move_delta is not None
    assert result.original_move_delta.move.move == chess.Move.from_uci("e2e4")
    assert result.modified_move_delta is not None
    assert result.modified_move_delta.move.move == chess.Move.from_uci("d2d4")


def test_automatic_sibling_selection_uses_uci_order_and_skips_unsatisfied_candidates():
    result = sibling_counterfactual(
        chess.Board(),
        chess.Move.from_uci("e2e4"),
        constraints=CounterfactualConstraints(
            changed_facts=(MaterialAnalyzer(FactPerspective.WHITE),),
        ),
    )

    assert not result.succeeded
    assert result.validity is CounterfactualValidity.NO_COUNTERFACTUAL
    assert result.failures[0].reason is CounterfactualFailureReason.NO_SATISFYING_COUNTERFACTUAL
    assert result.failures[0].message.startswith("None of 19 deterministic legal alternatives")
    assert all(failure.reason is CounterfactualFailureReason.CONSTRAINT_VIOLATION for failure in result.failures[1:])
    candidates = tuple(failure.candidate_move.uci() for failure in result.failures[1:])
    assert candidates == tuple(sorted(candidates))


def test_automatic_sibling_selection_returns_first_uci_alternative():
    result = sibling_counterfactual(chess.Board(), chess.Move.from_uci("e2e4"))

    assert result.succeeded
    assert isinstance(result.operator, SiblingMoveOperator)
    assert result.operator.alternative_move == chess.Move.from_uci("a2a3")


def test_structural_removal_reports_metadata_and_cannot_claim_reachability():
    constraints = CounterfactualConstraints(
        changed_attributes=frozenset((PositionAttribute.MATERIAL, PositionAttribute.CASTLING_RIGHTS)),
        preserved_attributes=frozenset(
            (
                PositionAttribute.TURN,
                PositionAttribute.KINGS,
                PositionAttribute.EN_PASSANT,
                PositionAttribute.HALFMOVE_CLOCK,
            )
        ),
        changed_facts=(MaterialAnalyzer(FactPerspective.WHITE),),
        preserved_facts=(MaterialAnalyzer(FactPerspective.BLACK),),
    )

    result = remove_piece_counterfactual(chess.Board(), chess.A1, constraints=constraints)

    assert result.succeeded
    assert result.validity is CounterfactualValidity.RULE_VALID
    assert result.modified is not None
    assert result.modified.rule_valid
    assert not result.modified.history_complete
    assert result.history.original_complete
    assert not result.history.modified_complete
    assert not result.history.shared_parent
    assert not result.history.legal_from_shared_parent
    assert not result.history.reachability_proven
    attributes = {item.attribute: item for item in result.attributes}
    assert len(attributes) == len(PositionAttribute)
    assert attributes[PositionAttribute.MATERIAL].changed
    assert attributes[PositionAttribute.CASTLING_RIGHTS].changed
    assert not attributes[PositionAttribute.TURN].changed
    assert {item.attribute for item in result.changed_attributes} == {
        PositionAttribute.MATERIAL,
        PositionAttribute.CASTLING_RIGHTS,
    }
    assert all(item.satisfied for item in (*result.changed_attributes, *result.preserved_attributes))
    assert result.changed_facts[0].relation is ConstraintRelation.CHANGED
    assert result.changed_facts[0].original.value == 39
    assert result.changed_facts[0].modified.value == 34
    assert result.preserved_facts[0].satisfied


def test_relocation_preserves_material_but_loses_structural_history():
    result = relocate_piece_counterfactual(
        chess.Board(),
        chess.B1,
        chess.A3,
        constraints=CounterfactualConstraints(
            changed_attributes=frozenset((PositionAttribute.HISTORY,)),
            preserved_attributes=frozenset(
                (
                    PositionAttribute.TURN,
                    PositionAttribute.KINGS,
                    PositionAttribute.MATERIAL,
                    PositionAttribute.CASTLING_RIGHTS,
                    PositionAttribute.EN_PASSANT,
                    PositionAttribute.HALFMOVE_CLOCK,
                )
            ),
            preserved_facts=(MaterialAnalyzer(FactPerspective.WHITE),),
        ),
    )

    assert result.succeeded
    assert result.modified is not None
    modified = chess.Board(result.modified.fen)
    assert modified.piece_at(chess.B1) is None
    assert modified.piece_at(chess.A3) == chess.Piece(chess.KNIGHT, chess.WHITE)
    assert all(item.satisfied for item in (*result.changed_attributes, *result.preserved_attributes))
    assert result.preserved_facts[0].satisfied


def test_structural_operators_return_structured_failures():
    empty = remove_piece_counterfactual(chess.Board(), chess.E4)
    king = remove_piece_counterfactual(chess.Board(), chess.E1)
    occupied = relocate_piece_counterfactual(chess.Board(), chess.B1, chess.A2)
    constraint = relocate_piece_counterfactual(
        chess.Board(),
        chess.B1,
        chess.A3,
        constraints=CounterfactualConstraints(
            changed_attributes=frozenset((PositionAttribute.MATERIAL,)),
        ),
    )

    assert empty.failures[0].reason is CounterfactualFailureReason.EMPTY_SOURCE_SQUARE
    assert king.failures[0].reason is CounterfactualFailureReason.KING_EDIT_FORBIDDEN
    assert occupied.failures[0].reason is CounterfactualFailureReason.OCCUPIED_TARGET_SQUARE
    assert constraint.failures[0].reason is CounterfactualFailureReason.CONSTRAINT_VIOLATION
    assert constraint.failures[0].attribute is PositionAttribute.MATERIAL
    assert all(result.modified is None for result in (empty, king, occupied, constraint))


def test_relocation_rejects_king_empty_source_and_invalid_result():
    empty = relocate_piece_counterfactual(chess.Board(), chess.E4, chess.E5)
    king = relocate_piece_counterfactual(chess.Board(), chess.E1, chess.E3)
    exposes_opposite_king = chess.Board("4k3/4b3/8/8/8/8/4R3/4K3 w - - 0 1")
    invalid = relocate_piece_counterfactual(exposes_opposite_king, chess.E7, chess.A7)

    assert empty.failures[0].reason is CounterfactualFailureReason.EMPTY_SOURCE_SQUARE
    assert king.failures[0].reason is CounterfactualFailureReason.KING_EDIT_FORBIDDEN
    assert invalid.failures[0].reason is CounterfactualFailureReason.INVALID_POSITION
    assert "status 1024" in invalid.failures[0].message


def test_structural_edit_clears_invalid_en_passant_and_reports_the_change():
    board = chess.Board()
    board.push_uci("e2e4")

    result = relocate_piece_counterfactual(
        board,
        chess.E4,
        chess.E5,
        constraints=CounterfactualConstraints(
            changed_attributes=frozenset((PositionAttribute.EN_PASSANT, PositionAttribute.HISTORY)),
        ),
    )

    assert result.succeeded
    assert result.modified is not None
    assert chess.Board(result.modified.fen).ep_square is None
    ep = next(item for item in result.attributes if item.attribute is PositionAttribute.EN_PASSANT)
    assert ep.original == chess.E3
    assert ep.modified is None
    assert ep.changed


def test_truncated_sibling_declares_only_shared_parent_legality():
    parent = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")

    result = sibling_counterfactual(
        parent,
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("d7d5"),
    )

    assert result.succeeded
    assert result.validity is CounterfactualValidity.SIBLING_LEGAL_MOVE
    assert result.history.shared_parent
    assert result.history.legal_from_shared_parent
    assert not result.history.original_complete
    assert not result.history.modified_complete
    assert not result.history.reachability_proven


def test_invalid_source_positions_and_malformed_configuration_are_rejected():
    invalid = chess.Board("8/8/8/8/8/8/4P3/4K3 w - - 0 1")

    sibling = sibling_counterfactual(invalid, chess.Move.from_uci("e2e3"))
    removal = remove_piece_counterfactual(invalid, chess.E2)
    relocation = relocate_piece_counterfactual(invalid, chess.E2, chess.E3)

    assert sibling.failures[0].reason is CounterfactualFailureReason.INVALID_POSITION
    assert not sibling.history.shared_parent
    assert not sibling.history.legal_from_shared_parent
    assert not sibling.history.reachability_proven
    assert removal.failures[0].reason is CounterfactualFailureReason.INVALID_POSITION
    assert relocation.failures[0].reason is CounterfactualFailureReason.INVALID_POSITION
    with pytest.raises(TypeError, match="python-chess Board"):
        remove_piece_counterfactual("not a board", chess.E2)
    with pytest.raises(ValueError, match="Invalid chess square"):
        RemovePieceOperator(99)
    with pytest.raises(ValueError, match="must differ"):
        RelocatePieceOperator(chess.A1, chess.A1)
    with pytest.raises(ValueError, match="both changed and preserved"):
        CounterfactualConstraints(
            changed_attributes=frozenset((PositionAttribute.TURN,)),
            preserved_attributes=frozenset((PositionAttribute.TURN,)),
        )
    with pytest.raises(ValueError, match="PositionAttribute"):
        CounterfactualConstraints(changed_attributes=frozenset(("turn",)))
    with pytest.raises(TypeError, match="FactAnalyzer"):
        CounterfactualConstraints(changed_facts=("material",))
    with pytest.raises(AssertionError, match="Unhandled position attribute"):
        counterfactuals._attribute_value(chess.Board(), object())


def test_sibling_failures_distinguish_illegal_moves_and_no_alternative():
    illegal_factual = sibling_counterfactual(chess.Board(), chess.Move.from_uci("e2e5"))
    illegal_alternative = sibling_counterfactual(
        chess.Board(), chess.Move.from_uci("e2e4"), chess.Move.from_uci("e2e4")
    )
    forced = chess.Board("8/8/8/8/8/6k1/5r2/7K w - - 0 1")
    only_move = next(iter(forced.legal_moves))
    no_alternative = sibling_counterfactual(forced, only_move)

    assert illegal_factual.failures[0].reason is CounterfactualFailureReason.ILLEGAL_FACTUAL_MOVE
    assert not illegal_factual.history.shared_parent
    assert not illegal_factual.history.legal_from_shared_parent
    assert illegal_alternative.failures[0].reason is CounterfactualFailureReason.ILLEGAL_ALTERNATIVE_MOVE
    assert len(illegal_alternative.failures) == 1
    assert no_alternative.failures[0].reason is CounterfactualFailureReason.NO_ALTERNATIVE


def test_property_every_explicit_starting_sibling_is_rule_valid_and_uses_same_parent():
    parent = chess.Board()
    factual = chess.Move.from_uci("e2e4")

    for alternative in sorted((move for move in parent.legal_moves if move != factual), key=lambda move: move.uci()):
        result = sibling_counterfactual(parent, factual, alternative)

        assert result.succeeded, alternative.uci()
        assert result.modified is not None
        assert result.modified.rule_valid
        assert result.history.shared_parent
        assert result.history.legal_from_shared_parent
        assert isinstance(result.operator, SiblingMoveOperator)
        assert result.operator.alternative_move == alternative


def test_property_removing_any_starting_non_king_piece_never_returns_a_malformed_state():
    board = chess.Board()

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        result = remove_piece_counterfactual(board, square)

        assert result.succeeded, chess.square_name(square)
        assert result.validity is CounterfactualValidity.RULE_VALID
        assert result.modified is not None
        assert result.modified.rule_valid
        assert not result.history.reachability_proven
