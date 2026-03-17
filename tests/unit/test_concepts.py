"""
Test cases for the concept module.
"""

import torch
from unittest.mock import MagicMock, patch

from lczerolens.concepts import (
    BinaryConcept,
    AndBinaryConcept,
    OrBinaryConcept,
    HasPiece,
    HasMaterialAdvantage,
    HasThreat,
    BestLegalMove,
    PieceBestLegalMove,
)
from lczerolens import LczeroBoard


class TestBinaryConcept:
    """
    Test cases for the BinaryConcept class.
    """

    def test_compute_metrics(self):
        """
        Test the compute_metrics method.
        """
        predictions = [0, 1, 0, 1]
        labels = [0, 1, 1, 1]
        metrics = BinaryConcept.compute_metrics(predictions, labels)
        assert metrics["accuracy"] == 0.75
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.6666666666666666

    def test_compute_label(self):
        """
        Test the compute_label method.
        """
        concept = AndBinaryConcept(HasPiece("p"), HasPiece("n"))
        assert concept.compute_label(LczeroBoard("8/8/8/8/8/8/8/8 w - - 0 1")) == 0
        assert concept.compute_label(LczeroBoard("8/p7/8/8/8/8/8/8 w - - 0 1")) == 0
        assert concept.compute_label(LczeroBoard("8/pn6/8/8/8/8/8/8 w - - 0 1")) == 1

    def test_relative_threat(self):
        """
        Test the relative threat concept.
        """
        concept = HasThreat("p", relative=True)
        assert concept.compute_label(LczeroBoard("8/8/8/8/8/8/8/8 w - - 0 1")) == 0
        assert concept.compute_label(LczeroBoard("R7/8/8/8/8/8/p7/8 w - - 0 1")) == 1
        assert concept.compute_label(LczeroBoard("R7/8/8/8/8/8/p7/8 b - - 0 1")) == 0

    def test_has_piece_relative_and_absolute(self):
        """Check HasPiece on trivially small boards."""
        board = LczeroBoard("8/8/8/8/8/8/P7/8 w - - 0 1")
        assert HasPiece("P", relative=True).compute_label(board) == 1
        assert HasPiece("p", relative=True).compute_label(board) == 0
        assert HasPiece("P", relative=False).compute_label(board) == 1
        board = LczeroBoard("8/8/8/8/8/8/P7/8 b - - 0 1")
        assert HasPiece("P", relative=True).compute_label(board) == 0
        assert HasPiece("p", relative=True).compute_label(board) == 1

    def test_or_and_binary_concepts(self):
        """Simple logical composition with tiny material positions."""
        has_white_pawn = HasPiece("P")
        has_black_knight = HasPiece("n", relative=False)
        concept_or = OrBinaryConcept(has_white_pawn, has_black_knight)
        concept_and = AndBinaryConcept(has_white_pawn, has_black_knight)

        b_empty = LczeroBoard("8/8/8/8/8/8/8/8 w - - 0 1")
        b_wp = LczeroBoard("8/8/8/8/8/8/P7/8 w - - 0 1")
        b_wp_bn = LczeroBoard("8/8/8/8/8/8/P7/1n6 w - - 0 1")

        assert concept_or.compute_label(b_empty) == 0
        assert concept_or.compute_label(b_wp) == 1
        assert concept_or.compute_label(b_wp_bn) == 1

        assert concept_and.compute_label(b_empty) == 0
        assert concept_and.compute_label(b_wp) == 0
        assert concept_and.compute_label(b_wp_bn) == 1

    def test_has_material_advantage_relative_and_absolute(self):
        """Material advantage on boards with one or two pieces."""
        board = LczeroBoard("k7/8/8/8/8/8/P7/7K w - - 0 1")
        assert HasMaterialAdvantage(relative=True).compute_label(board) == 1
        assert HasMaterialAdvantage(relative=False).compute_label(board) == 1
        board = LczeroBoard("k7/8/8/8/8/8/P7/7K b - - 0 1")
        assert HasMaterialAdvantage(relative=True).compute_label(board) == 0
        assert HasMaterialAdvantage(relative=False).compute_label(board) == 1


class TestMoveConcepts:
    """Test cases for BestLegalMove and PieceBestLegalMove."""

    def test_best_legal_move(self):
        """Test BestLegalMove returns the index of the policy's best legal move."""
        board = LczeroBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        legal_indices = [LczeroBoard.encode_move(move, board.turn) for move in board.legal_moves]

        policy = torch.zeros(1858)
        policy[legal_indices[0]] = 10.0
        policy[legal_indices[1:]] = -10.0

        mock_policy_flow = MagicMock(return_value={"policy": policy.unsqueeze(0)})
        mock_model = MagicMock()
        mock_model.module = MagicMock()

        with patch("lczerolens.concepts.PolicyFlow") as mock_policy_flow_cls:
            mock_policy_flow_cls.from_model.return_value = mock_policy_flow

            concept = BestLegalMove(mock_model)
            label = concept.compute_label(board)

        assert label == legal_indices[0]
        mock_policy_flow_cls.from_model.assert_called_once_with(mock_model.module)
        mock_policy_flow.assert_called_once_with(board)

    def test_piece_best_legal_move(self):
        """Test PieceBestLegalMove returns 1 when best move is from the specified piece."""
        board = LczeroBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        legal_moves = list(board.legal_moves)
        legal_indices = [LczeroBoard.encode_move(move, board.turn) for move in legal_moves]

        e2e4 = next(m for m in legal_moves if m.uci() == "e2e4")
        e2e4_idx = LczeroBoard.encode_move(e2e4, board.turn)

        policy = torch.zeros(1858)
        policy[e2e4_idx] = 10.0
        for idx in legal_indices:
            if idx != e2e4_idx:
                policy[idx] = -10.0

        mock_policy_flow = MagicMock(return_value={"policy": policy.unsqueeze(0)})
        mock_model = MagicMock()
        mock_model.module = MagicMock()

        with patch("lczerolens.concepts.PolicyFlow") as mock_policy_flow_cls:
            mock_policy_flow_cls.from_model.return_value = mock_policy_flow

            concept = PieceBestLegalMove(mock_model, "P")
            label = concept.compute_label(board)

        assert label == 1
        mock_policy_flow_cls.from_model.assert_called_once_with(mock_model.module)

    def test_piece_best_legal_move_wrong_piece(self):
        """Test PieceBestLegalMove returns 0 when best move is not from the specified piece."""
        board = LczeroBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        legal_moves = list(board.legal_moves)
        legal_indices = [LczeroBoard.encode_move(move, board.turn) for move in legal_moves]

        g1f3 = next(m for m in legal_moves if m.uci() == "g1f3")
        g1f3_idx = LczeroBoard.encode_move(g1f3, board.turn)

        policy = torch.zeros(1858)
        policy[g1f3_idx] = 10.0
        for idx in legal_indices:
            if idx != g1f3_idx:
                policy[idx] = -10.0

        mock_policy_flow = MagicMock(return_value={"policy": policy.unsqueeze(0)})
        mock_model = MagicMock()
        mock_model.module = MagicMock()

        with patch("lczerolens.concepts.PolicyFlow") as mock_policy_flow_cls:
            mock_policy_flow_cls.from_model.return_value = mock_policy_flow

            concept = PieceBestLegalMove(mock_model, "P")
            label = concept.compute_label(board)

        assert label == 0
