"""Authored chess tasks and deterministic puzzle grading.

Puzzle correctness is normative source evidence: it says which continuations an
author accepts.  It is deliberately separate from evaluator preference, search
strength, and exact move analysis, which remain observational evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import chess

from .provenance import ChessPlayer, PositionIdentity


class PuzzleStatus(str, Enum):
    """Outcome of comparing an attempted continuation with an authored solution."""

    IN_PROGRESS = "in_progress"
    SOLVED = "solved"
    FAILED = "failed"


@dataclass(frozen=True)
class PuzzleProvenance:
    """Stable identity assigned by the source that authored a puzzle."""

    source: str
    identifier: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Puzzle provenance source must not be empty.")
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError("Puzzle provenance identifier must not be empty.")


@dataclass(frozen=True)
class PuzzleContinuation:
    """One accepted move and the authored continuations after it.

    Siblings are alternative accepted moves.  Children are the moves accepted
    at the next ply.  A leaf means that the puzzle is solved after this move,
    even when the resulting chess position is not terminal.
    """

    move: str
    continuations: tuple[PuzzleContinuation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.move, str):
            raise ValueError("Puzzle continuation moves must be UCI strings.")
        try:
            chess.Move.from_uci(self.move)
        except ValueError as error:
            raise ValueError("Puzzle continuation moves must be valid UCI strings.") from error
        if not isinstance(self.continuations, tuple) or any(
            not isinstance(item, PuzzleContinuation) for item in self.continuations
        ):
            raise ValueError("Puzzle continuations must be a tuple of PuzzleContinuation values.")
        moves = tuple(item.move for item in self.continuations)
        if len(moves) != len(set(moves)):
            raise ValueError("Alternative puzzle continuations must have unique moves.")


@dataclass(frozen=True)
class PuzzleSolution:
    """A non-empty tree of source-authored accepted continuations."""

    continuations: tuple[PuzzleContinuation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.continuations, tuple) or not self.continuations:
            raise ValueError("Puzzle solutions need at least one continuation.")
        if any(not isinstance(item, PuzzleContinuation) for item in self.continuations):
            raise ValueError("Puzzle solutions require PuzzleContinuation values.")
        moves = tuple(item.move for item in self.continuations)
        if len(moves) != len(set(moves)):
            raise ValueError("Alternative puzzle solution moves must be unique.")


@dataclass(frozen=True)
class Puzzle:
    """A reconstructable position paired with an authored solution strategy."""

    position: PositionIdentity
    solution: PuzzleSolution
    provenance: PuzzleProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.position, PositionIdentity):
            raise ValueError("Puzzles require a PositionIdentity.")
        if not isinstance(self.solution, PuzzleSolution):
            raise ValueError("Puzzles require a PuzzleSolution.")
        if self.provenance is not None and not isinstance(self.provenance, PuzzleProvenance):
            raise ValueError("Puzzle provenance must be PuzzleProvenance when present.")
        board = self.position.board()
        _validate_continuations(board, self.solution.continuations, self.solver, ply=0)

    @classmethod
    def from_board(
        cls,
        board: chess.Board,
        solution: PuzzleSolution,
        *,
        provenance: PuzzleProvenance | None = None,
    ) -> Puzzle:
        """Freeze a board and its retained history as an authored puzzle."""
        return cls(PositionIdentity.from_board(board), solution, provenance)

    @property
    def solver(self) -> ChessPlayer:
        """The absolute player who makes the first authored move."""
        return self.position.player

    def accepted_moves(self, moves: Iterable[chess.Move | str] = ()) -> tuple[chess.Move, ...]:
        """Return the source-authored moves accepted after an attempted prefix."""
        attempted = _canonical_moves(moves)
        status, _, continuations = _grade(self.solution.continuations, attempted)
        if status is PuzzleStatus.FAILED:
            return ()
        return tuple(chess.Move.from_uci(item.move) for item in continuations)

    def grade(self, moves: Iterable[chess.Move | str]) -> PuzzleAttempt:
        """Compare a full-ply attempted prefix with the authored solution tree."""
        attempted = _canonical_moves(moves)
        status, failure_ply, _ = _grade(self.solution.continuations, attempted)
        return PuzzleAttempt(self, attempted, status, failure_ply)


@dataclass(frozen=True)
class PuzzleAttempt:
    """Immutable result of grading a full-ply prefix against one puzzle."""

    puzzle: Puzzle
    moves: tuple[str, ...]
    status: PuzzleStatus
    failure_ply: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.puzzle, Puzzle):
            raise ValueError("Puzzle attempts require a Puzzle.")
        if not isinstance(self.moves, tuple) or any(not isinstance(move, str) for move in self.moves):
            raise ValueError("Puzzle attempt moves must be a tuple of UCI strings.")
        try:
            _canonical_moves(self.moves)
        except ValueError as error:
            raise ValueError("Puzzle attempt moves must be a tuple of UCI strings.") from error
        if not isinstance(self.status, PuzzleStatus):
            raise ValueError("Puzzle attempt status must be a PuzzleStatus.")
        expected_status, expected_failure, _ = _grade(self.puzzle.solution.continuations, self.moves)
        if self.status is not expected_status or self.failure_ply != expected_failure:
            raise ValueError("Puzzle attempt status must match its moves and puzzle solution.")

    @property
    def accepted_moves(self) -> tuple[chess.Move, ...]:
        """Return the authored moves that can continue this attempt."""
        return self.puzzle.accepted_moves(self.moves)


def _validate_continuations(
    board: chess.Board,
    continuations: tuple[PuzzleContinuation, ...],
    solver: ChessPlayer,
    *,
    ply: int,
) -> None:
    for continuation in continuations:
        try:
            move = board.parse_uci(continuation.move)
        except ValueError as error:
            raise ValueError(f"Puzzle solution move {continuation.move!r} is illegal at ply {ply}.") from error
        after = board.copy(stack=True)
        after.push(move)
        if continuation.continuations:
            if after.is_game_over():
                raise ValueError("Puzzle solutions cannot continue after a terminal position.")
            _validate_continuations(after, continuation.continuations, solver, ply=ply + 1)
        elif ChessPlayer.from_color(board.turn) is not solver:
            raise ValueError("Puzzle solution leaves must follow a solver move.")


def _canonical_moves(moves: Iterable[chess.Move | str]) -> tuple[str, ...]:
    attempted: list[str] = []
    for move in moves:
        if isinstance(move, chess.Move):
            attempted.append(move.uci())
            continue
        if not isinstance(move, str):
            raise ValueError("Puzzle attempt moves must be chess.Move values or UCI strings.")
        try:
            chess.Move.from_uci(move)
        except ValueError as error:
            raise ValueError("Puzzle attempt moves must be valid UCI strings.") from error
        attempted.append(move)
    return tuple(attempted)


def _grade(
    continuations: tuple[PuzzleContinuation, ...],
    moves: tuple[str, ...],
) -> tuple[PuzzleStatus, int | None, tuple[PuzzleContinuation, ...]]:
    available = continuations
    for ply, move in enumerate(moves):
        selected = next((item for item in available if item.move == move), None)
        if selected is None:
            return PuzzleStatus.FAILED, ply, ()
        available = selected.continuations
        if not available and ply != len(moves) - 1:
            return PuzzleStatus.FAILED, ply + 1, ()
    if moves and not available:
        return PuzzleStatus.SOLVED, None, ()
    return PuzzleStatus.IN_PROGRESS, None, available


__all__ = [
    "Puzzle",
    "PuzzleAttempt",
    "PuzzleContinuation",
    "PuzzleProvenance",
    "PuzzleSolution",
    "PuzzleStatus",
]
