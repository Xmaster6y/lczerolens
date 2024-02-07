"""
All concepts related to material.
"""

import chess

from lczerolens.xai.concept import BinaryConcept


class HasPieceConcept(BinaryConcept):
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
        board: chess.Board,
    ) -> int:
        """
        Compute the label for a given model and input.
        """
        if self.relative:
            color = self.piece.color and board.turn
        else:
            color = self.piece.color
        squares = board.pieces(self.piece.piece_type, color)
        return len(squares) > 0
