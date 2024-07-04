"""Puzzle tests."""

import pytest


from lczerolens.play.puzzle import Puzzle
from lczerolens.play.sampling import RandomSampler


@pytest.fixture
def easy_puzzle():
    return {
        "PuzzleId": "1",
        "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Moves": "e2e4 e7e5",
        "Rating": 1000,
        "RatingDeviation": 100,
        "Popularity": 1000,
        "NbPlays": 1000,
        "Themes": "Opening",
        "GameUrl": "https://lichess.org/training/1",
        "OpeningTags": "Ruy Lopez",
    }


class TestRandomSampler:
    def test_puzzle_evaluation(self, easy_puzzle):
        """Test puzzle evaluation."""
        puzzle = Puzzle.from_dict(easy_puzzle)
        sampler = RandomSampler(use_argmax=True)
        score, perplexity = puzzle.evaluate(sampler, use_perplexity=True)
        assert score == 0.0
        assert abs(perplexity - 20.0) < 1e-3
