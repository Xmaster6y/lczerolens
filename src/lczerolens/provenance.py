"""Stable identities and producer provenance for persisted analysis evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

import chess
import chess.variant


class ChessPlayer(str, Enum):
    """An absolute chess player."""

    WHITE = "white"
    BLACK = "black"

    @classmethod
    def from_color(cls, color: chess.Color) -> "ChessPlayer":
        """Convert a python-chess color to its stable evidence value."""
        return cls.WHITE if color is chess.WHITE else cls.BLACK


@dataclass(frozen=True)
class PositionIdentity:
    """A variant-aware position and the retained history that reconstructs it."""

    fen: str
    start_fen: str
    moves: tuple[str, ...]
    variant: str = "chess"
    chess960: bool = False

    def __post_init__(self) -> None:
        if not self.variant:
            raise ValueError("Position variant must not be empty.")
        if not isinstance(self.chess960, bool):
            raise ValueError("chess960 must be a boolean.")
        try:
            reconstructed = self._new_board(self.start_fen)
            for move_uci in self.moves:
                move = reconstructed.parse_uci(move_uci)
                reconstructed.push(move)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Position history must be a legal sequence from start_fen.") from error
        if reconstructed.fen() != self.fen:
            raise ValueError("Position history must reconstruct fen.")

    @classmethod
    def from_board(cls, board: chess.Board) -> "PositionIdentity":
        """Freeze a board without discarding its retained move stack or variant."""
        if not isinstance(board, chess.Board):
            raise TypeError("PositionIdentity requires a python-chess Board.")
        root = board.root()
        return cls(
            fen=board.fen(),
            start_fen=root.fen(),
            moves=tuple(move.uci() for move in board.move_stack),
            variant=board.uci_variant,
            chess960=board.chess960,
        )

    @property
    def player(self) -> ChessPlayer:
        """Absolute side to move in the recorded position."""
        return ChessPlayer.from_color(self.board().turn)

    def board(self) -> chess.Board:
        """Reconstruct a defensive board with the retained history."""
        board = self._new_board(self.start_fen)
        for move_uci in self.moves:
            board.push_uci(move_uci)
        return board

    def _new_board(self, fen: str) -> chess.Board:
        try:
            board_type = chess.variant.find_variant(self.variant)
        except ValueError as error:
            raise ValueError(f"Unsupported chess variant {self.variant!r}.") from error
        return board_type(fen, chess960=self.chess960)


@dataclass(frozen=True)
class EvaluationProvenance:
    """Identity of the evaluator and network that produced an evaluation."""

    source: str
    model_type: str
    network: str | None = None
    network_checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Evaluation provenance source must not be empty.")
        if not self.model_type:
            raise ValueError("Evaluation provenance model_type must not be empty.")
        if self.network_checksum is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", self.network_checksum):
            raise ValueError("network_checksum must be a lowercase sha256 digest.")


__all__ = ["ChessPlayer", "EvaluationProvenance", "PositionIdentity"]
