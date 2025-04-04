"""
Test cases for the concept module.
"""

from lczerolens.concept import BinaryConcept, AndBinaryConcept
from lczerolens.concepts import (
    HasPiece,
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
