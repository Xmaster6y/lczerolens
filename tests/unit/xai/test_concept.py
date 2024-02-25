"""
Test cases for the concept module.
"""

import chess

from lczerolens import BoardDataset, GameDataset
from lczerolens.xai import (
    AndBinaryConcept,
    BinaryConcept,
    ConceptDataset,
    HasPieceConcept,
    HasThreatConcept,
)


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
        concept = AndBinaryConcept(HasPieceConcept("p"), HasPieceConcept("n"))
        assert (
            concept.compute_label(chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"))
            == 0
        )
        assert (
            concept.compute_label(chess.Board("8/p7/8/8/8/8/8/8 w - - 0 1"))
            == 0
        )
        assert (
            concept.compute_label(chess.Board("8/pn6/8/8/8/8/8/8 w - - 0 1"))
            == 1
        )

    def test_relative_threat(self):
        """
        Test the relative threat concept.
        """
        concept = HasThreatConcept(
            "p", relative=True
        )  # Is an enemy pawn threatened?
        assert (
            concept.compute_label(chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"))
            == 0
        )
        assert (
            concept.compute_label(chess.Board("R7/8/8/8/8/8/p7/8 w - - 0 1"))
            == 1
        )
        assert (
            concept.compute_label(chess.Board("R7/8/8/8/8/8/p7/8 b - - 0 1"))
            == 0
        )


class TestDataset:
    def test_unique_concept_dataset(self):
        dataset = ConceptDataset(
            concept=HasPieceConcept("p"),
            boards=[
                chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"),
                chess.Board("8/8/8/8/8/8/8/8 w - - 0 1"),
            ],
            labels=[0, 0],
        )
        dataset.filter_unique_()
        assert len(dataset) == 1

    def test_conversion(self, game_dataset_10: GameDataset):
        board_dataset = BoardDataset.from_game_dataset(game_dataset_10)
        concept_dataset = ConceptDataset.from_game_dataset(game_dataset_10)
        concept_dataset.filter_unique_()
        concept_dataset.set_concept(HasPieceConcept("p"))
        fen_set = set([board.fen() for _, board in board_dataset])
        assert len(concept_dataset.boards) == len(fen_set)
        assert len(concept_dataset[0][1].move_stack) == 0
