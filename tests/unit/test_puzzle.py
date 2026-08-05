"""Tests for source-authored puzzle tasks and grading."""

from dataclasses import replace

import chess
import pytest

from lczerolens import (
    Puzzle,
    PuzzleAttempt,
    PuzzleContinuation,
    PuzzleProvenance,
    PuzzleSolution,
    PuzzleStatus,
)


def test_puzzle_freezes_position_and_grades_authored_completion_not_terminality():
    board = chess.Board("8/8/8/8/8/4k3/8/4K2R w - - 0 1")
    puzzle = Puzzle.from_board(
        board,
        PuzzleSolution((PuzzleContinuation("h1h2"),)),
        provenance=PuzzleProvenance("fixture", "non-terminal-goal"),
    )

    board.push_uci("h1h2")
    attempt = puzzle.grade(("h1h2",))

    assert puzzle.position.board().fen() != board.fen()
    assert puzzle.solver.value == "white"
    assert attempt.status is PuzzleStatus.SOLVED
    assert not board.is_game_over()
    assert not attempt.accepted_moves


def test_solution_tree_exposes_alternatives_and_grades_full_ply_prefixes():
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    solution = PuzzleSolution(
        (
            PuzzleContinuation("g6g7"),
            PuzzleContinuation("g6h6", (PuzzleContinuation("h8g8", (PuzzleContinuation("h6g7"),)),)),
        )
    )
    puzzle = Puzzle.from_board(board, solution)

    assert tuple(move.uci() for move in puzzle.accepted_moves()) == ("g6g7", "g6h6")
    assert puzzle.grade(()).status is PuzzleStatus.IN_PROGRESS
    assert puzzle.grade((chess.Move.from_uci("g6g7"),)).status is PuzzleStatus.SOLVED

    partial = puzzle.grade(("g6h6",))
    assert partial.status is PuzzleStatus.IN_PROGRESS
    assert tuple(move.uci() for move in partial.accepted_moves) == ("h8g8",)
    assert puzzle.grade(("g6h6", "h8g8", "h6g7")).status is PuzzleStatus.SOLVED

    failed = puzzle.grade(("g6h6", "h8g8", "h6f8"))
    assert failed.status is PuzzleStatus.FAILED
    assert failed.failure_ply == 2
    assert not failed.accepted_moves


def test_puzzle_rejects_malformed_inconsistent_and_illegal_solution_trees():
    with pytest.raises(ValueError, match="UCI strings"):
        PuzzleContinuation(1)
    with pytest.raises(ValueError, match="valid UCI"):
        PuzzleContinuation("not-a-move")
    with pytest.raises(ValueError, match="tuple of PuzzleContinuation"):
        PuzzleContinuation("e2e4", [])
    with pytest.raises(ValueError, match="unique moves"):
        PuzzleContinuation("e2e4", (PuzzleContinuation("e7e5"), PuzzleContinuation("e7e5")))
    with pytest.raises(ValueError, match="at least one"):
        PuzzleSolution(())
    with pytest.raises(ValueError, match="PuzzleContinuation values"):
        PuzzleSolution((object(),))
    with pytest.raises(ValueError, match="unique"):
        PuzzleSolution((PuzzleContinuation("e2e4"), PuzzleContinuation("e2e4")))
    with pytest.raises(ValueError, match="illegal at ply 0"):
        Puzzle.from_board(chess.Board(), PuzzleSolution((PuzzleContinuation("e7e5"),)))

    response_leaf = PuzzleSolution((PuzzleContinuation("e2e4", (PuzzleContinuation("e7e5"),)),))
    with pytest.raises(ValueError, match="leaves must follow a solver move"):
        Puzzle.from_board(chess.Board(), response_leaf)

    after_mate = PuzzleSolution(
        (
            PuzzleContinuation(
                "g6g7",
                (PuzzleContinuation("h8g8"),),
            ),
        )
    )
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    with pytest.raises(ValueError, match="terminal position"):
        Puzzle.from_board(board, after_mate)


def test_attempts_are_self_validating_and_extra_moves_fail_after_completion():
    board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    puzzle = Puzzle.from_board(board, PuzzleSolution((PuzzleContinuation("g6g7"),)))

    extra = puzzle.grade(("g6g7", "h8g8"))
    assert extra.status is PuzzleStatus.FAILED
    assert extra.failure_ply == 1
    with pytest.raises(ValueError, match="must match"):
        PuzzleAttempt(puzzle, ("g6g7",), PuzzleStatus.FAILED, 0)
    with pytest.raises(ValueError, match="require a Puzzle"):
        PuzzleAttempt("puzzle", (), PuzzleStatus.IN_PROGRESS)
    with pytest.raises(ValueError, match="tuple of UCI"):
        PuzzleAttempt(puzzle, [], PuzzleStatus.IN_PROGRESS)
    with pytest.raises(ValueError, match="valid UCI"):
        puzzle.grade(("bad",))
    with pytest.raises(ValueError, match="chess.Move"):
        puzzle.grade((object(),))


def test_record_validation_rejects_wrong_value_types():
    continuation = PuzzleContinuation("e2e4")
    solution = PuzzleSolution((continuation,))
    puzzle = Puzzle.from_board(chess.Board(), solution)

    with pytest.raises(ValueError, match="source"):
        PuzzleProvenance("", "id")
    with pytest.raises(ValueError, match="identifier"):
        PuzzleProvenance("source", "")
    with pytest.raises(ValueError, match="PositionIdentity"):
        Puzzle("position", solution)
    with pytest.raises(ValueError, match="PuzzleSolution"):
        Puzzle(puzzle.position, "solution")
    with pytest.raises(ValueError, match="PuzzleProvenance"):
        Puzzle(puzzle.position, solution, "source")
    with pytest.raises(ValueError, match="PuzzleStatus"):
        replace(puzzle.grade(()), status="in_progress")
    with pytest.raises(ValueError, match="tuple of UCI"):
        PuzzleAttempt(puzzle, ("bad",), PuzzleStatus.FAILED, 0)
