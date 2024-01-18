"""
Tests for the move utils.
"""

import chess

from lczerolens import move_utils


class TestStability:
    def test_encode_decode(self, random_move_board_list):
        """
        Test that encoding and decoding a move is the identity.
        """
        us, them = chess.WHITE, chess.BLACK
        for move, board in zip(*random_move_board_list):
            encoded_move = move_utils.encode_move(move, (us, them))
            decoded_move = move_utils.decode_move(
                encoded_move, (us, them), board
            )
            assert move == decoded_move
            us, them = them, us
