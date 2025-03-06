"""All concepts related to threats."""

import chess

from lczerolens.board import LczeroBoard
from lczerolens.concept import BinaryConcept


class HasThreat(BinaryConcept):
    """
    Class for material concept-based XAI methods.
    """

    def __init__(
        self,
        piece: str,
        relative: bool = True,
    ):
        """
        Initialize the class.
        """
        self.piece = chess.Piece.from_symbol(piece)
        self.relative = relative

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """
        Compute the label for a given model and input.
        """
        if self.relative:
            color = self.piece.color if board.turn else not self.piece.color
        else:
            color = self.piece.color
        squares = board.pieces(self.piece.piece_type, color)
        for square in squares:
            if board.is_attacked_by(not color, square):
                return 1
        return 0


class HasMateThreat(BinaryConcept):
    """
    Class for material concept-based XAI methods.
    """

    def compute_label(
        self,
        board: LczeroBoard,
    ) -> int:
        """
        Compute the label for a given model and input.
        """
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return 1
            board.pop()
        return 0
