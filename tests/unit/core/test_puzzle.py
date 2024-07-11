"""Puzzle tests."""

import pytest


from lczerolens.play.puzzle import Puzzle
from lczerolens.play.sampling import RandomSampler, PolicySampler


@pytest.fixture
def opening_puzzle():
    return {
        "PuzzleId": "1",
        "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Moves": "e2e4 e7e5 d2d4 d7d5",
        "Rating": 1000,
        "RatingDeviation": 100,
        "Popularity": 1000,
        "NbPlays": 1000,
        "Themes": "Opening",
        "GameUrl": "https://lichess.org/training/1",
        "OpeningTags": "Ruy Lopez",
    }


@pytest.fixture
def easy_puzzle():
    return {
        "PuzzleId": "1",
        "FEN": "1r5r/p3kp2/4p2p/4P3/3R1Pp1/6P1/P1P4P/4K2R w K - 1 25",
        "Moves": "d4a4 b8b1 e1f2 b1h1 a4a7 e7f8",
        "Rating": 1000,
        "RatingDeviation": 100,
        "Popularity": 1000,
        "NbPlays": 1000,
        "Themes": "Opening",
        "GameUrl": "https://lichess.org/training/1",
        "OpeningTags": "Ruy Lopez",
    }


class TestPuzzle:
    def test_puzzle_creation(self, opening_puzzle):
        """Test puzzle creation."""
        puzzle = Puzzle.from_dict(opening_puzzle)
        assert len(puzzle) == 3
        assert puzzle.rating == 1000
        assert puzzle.rating_deviation == 100
        assert puzzle.popularity == 1000
        assert puzzle.nb_plays == 1000
        assert puzzle.themes == ["Opening"]
        assert puzzle.game_url == "https://lichess.org/training/1"
        assert puzzle.opening_tags == ["Ruy", "Lopez"]

    def test_puzzle_use(self, opening_puzzle):
        """Test puzzle use."""
        puzzle = Puzzle.from_dict(opening_puzzle)
        assert len(list(puzzle.board_move_generator())) == 2
        assert len(list(puzzle.board_move_generator(all_moves=True))) == 3


class TestRandomSampler:
    def test_puzzle_evaluation(self, opening_puzzle):
        """Test puzzle evaluation."""
        puzzle = Puzzle.from_dict(opening_puzzle)
        sampler = RandomSampler(use_argmax=True)
        metrics = puzzle.evaluate(sampler)
        assert metrics["score"] == 0.0
        assert abs(metrics["perplexity"] - (20.0 * 30) ** 0.5) < 1e-3

    def test_puzzle_multiple_evaluation_len(self, easy_puzzle):
        """Test puzzle evaluation."""
        puzzles = [Puzzle.from_dict(easy_puzzle) for _ in range(10)]
        sampler = RandomSampler()
        all_results = Puzzle.evaluate_multiple(puzzles, sampler, all_moves=True, compute_metrics=False)
        assert len(list(all_results)) == 10 * 5
        results = Puzzle.evaluate_multiple(puzzles, sampler, compute_metrics=False)
        assert len(list(results)) == 10 * 3

    def test_puzzle_multiple_evaluation(self, easy_puzzle):
        """Test puzzle evaluation."""
        puzzles = [Puzzle.from_dict(easy_puzzle) for _ in range(10)]
        sampler = RandomSampler()
        all_results = Puzzle.evaluate_multiple(puzzles, sampler, all_moves=True)
        assert len(list(all_results)) == 10
        results = Puzzle.evaluate_multiple(puzzles, sampler, all_moves=False)
        assert len(list(results)) == 10


class TestPolicySampler:
    def test_puzzle_evaluation(self, easy_puzzle, winner_model):
        """Test puzzle evaluation."""
        puzzle = Puzzle.from_dict(easy_puzzle)
        sampler = PolicySampler(model=winner_model, use_argmax=True)
        metrics = puzzle.evaluate(sampler, all_moves=True)
        assert metrics["score"] > 0.0
        assert metrics["perplexity"] < 15.0

    def test_puzzle_multiple_evaluation(self, easy_puzzle, tiny_model):
        """Test puzzle evaluation."""
        puzzles = [Puzzle.from_dict(easy_puzzle) for _ in range(10)]
        sampler = PolicySampler(model=tiny_model, use_argmax=False)
        all_results = Puzzle.evaluate_multiple(puzzles, sampler, all_moves=True)
        assert len(list(all_results)) == 10
        results = Puzzle.evaluate_multiple(puzzles, sampler, all_moves=False)
        assert len(list(results)) == 10
