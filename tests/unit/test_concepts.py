"""
Test cases for the concept module.
"""

from lczerolens.concepts import (
    BinaryConcept,
    AndBinaryConcept,
    OrBinaryConcept,
    HasPiece,
    HasMaterialAdvantage,
    HasThreat,
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
        concept = HasThreat("p", relative=True)  # Is an enemy pawn threatened?
        assert concept.compute_label(LczeroBoard("8/8/8/8/8/8/8/8 w - - 0 1")) == 0
        assert concept.compute_label(LczeroBoard("R7/8/8/8/8/8/p7/8 w - - 0 1")) == 1
        assert concept.compute_label(LczeroBoard("R7/8/8/8/8/8/p7/8 b - - 0 1")) == 0

    def test_has_piece_relative_and_absolute(self):
        """Check HasPiece on trivially small boards."""
        # One white pawn on a2, white to move
        board = LczeroBoard("8/8/8/8/8/8/P7/8 w - - 0 1")
        assert HasPiece("P", relative=True).compute_label(board) == 1
        assert HasPiece("p", relative=True).compute_label(board) == 0
        # Absolute: white pawn exists regardless of turn
        assert HasPiece("P", relative=False).compute_label(board) == 1
        # Switch turn
        board = LczeroBoard("8/8/8/8/8/8/P7/8 b - - 0 1")
        # Relative now refers to black's perspective
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
        # White to move, one white pawn vs lone black king → advantage
        board = LczeroBoard("k7/8/8/8/8/8/P7/7K w - - 0 1")
        assert HasMaterialAdvantage(relative=True).compute_label(board) == 1
        assert HasMaterialAdvantage(relative=False).compute_label(board) == 1
        # Black to move, same position but turn flips relative perspective
        board = LczeroBoard("k7/8/8/8/8/8/P7/7K b - - 0 1")
        # Relative: side to move is black (no material) → not advantage
        assert HasMaterialAdvantage(relative=True).compute_label(board) == 0
        # Absolute advantage for white remains
        assert HasMaterialAdvantage(relative=False).compute_label(board) == 1
