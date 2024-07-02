"""
Tests for the move utils.
"""

import chess
from lczero.backends import GameState

from lczerolens import move_encodings
from lczerolens.encodings import backends as lczero_utils


class TestStability:
    def test_encode_decode(self, random_move_board_list):
        """
        Test that encoding and decoding a move is the identity.
        """
        us, them = chess.WHITE, chess.BLACK
        for move, board in zip(*random_move_board_list):
            encoded_move = move_encodings.encode_move(move, us)
            decoded_move = move_encodings.decode_move(encoded_move, board)
            assert move == decoded_move
            us, them = them, us


class TestBackend:
    def test_encode_decode_random(self, random_move_board_list):
        """
        Test that encoding and decoding a move corresponds to the backend.
        """
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            legal_moves = [move.uci() for move in board.legal_moves]
            (
                lczero_legal_moves,
                lczero_policy_indices,
            ) = lczero_utils.moves_with_castling_swap(lczero_game, board)
            assert len(legal_moves) == len(lczero_legal_moves)
            assert set(legal_moves) == set(lczero_legal_moves)
            policy_indices = [move_encodings.encode_move(move, board.turn) for move in board.legal_moves]
            assert len(lczero_policy_indices) == len(policy_indices)
            assert set(lczero_policy_indices) == set(policy_indices)

    def test_encode_decode_long(self, long_move_board_list):
        """
        Test that encoding and decoding a move corresponds to the backend.
        """
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            legal_moves = [move.uci() for move in board.legal_moves]
            (
                lczero_legal_moves,
                lczero_policy_indices,
            ) = lczero_utils.moves_with_castling_swap(lczero_game, board)
            assert len(legal_moves) == len(lczero_legal_moves)
            assert set(legal_moves) == set(lczero_legal_moves)
            policy_indices = [move_encodings.encode_move(move, board.turn) for move in board.legal_moves]
            assert len(lczero_policy_indices) == len(policy_indices)
            assert set(lczero_policy_indices) == set(policy_indices)
