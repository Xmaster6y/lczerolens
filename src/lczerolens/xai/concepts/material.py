"""All concepts related to material."""

from typing import Dict, Optional

import chess

from lczerolens.xai.concept import BinaryConcept


class HasPieceConcept(BinaryConcept):
    """Class for material concept-based XAI methods."""

    def __init__(
        self,
        piece: str,
        relative: bool = True,
    ):
        """Initialize the class."""
        self.piece = chess.Piece.from_symbol(piece)
        self.relative = relative

    def compute_label(
        self,
        board: chess.Board,
    ) -> int:
        """Compute the label for a given model and input."""
        if self.relative:
            color = self.piece.color if board.turn else not self.piece.color
        else:
            color = self.piece.color
        squares = board.pieces(self.piece.piece_type, color)
        return 1 if len(squares) > 0 else 0


class HasMaterialAdvantageConcept(BinaryConcept):
    """Class for material concept-based XAI methods.

    Attributes
    ----------
    piece_values : Dict[int, int]
        The piece values.
    """

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(
        self,
        relative: bool = True,
    ):
        """
        Initialize the class.
        """
        self.relative = relative

    def compute_label(
        self,
        board: chess.Board,
        piece_values: Optional[Dict[int, int]] = None,
    ) -> int:
        """
        Compute the label for a given model and input.
        """
        if piece_values is None:
            piece_values = self.piece_values
        if self.relative:
            us, them = board.turn, not board.turn
        else:
            us, them = chess.WHITE, chess.BLACK
        our_value = 0
        their_value = 0
        for piece in range(1, 7):
            our_value += len(board.pieces(piece, us)) * piece_values[piece]
            their_value += len(board.pieces(piece, them)) * piece_values[piece]
        return 1 if our_value > their_value else 0
