"""
Tests for the board utils.
"""

import sys
from typing import List, Tuple
import pytest
import chess
from lczero.backends import GameState

from lczerolens import backends as lczero_utils
from lczerolens import LczeroBoard


class TestWithBackend:
    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
    def test_board_to_config_tensor(
        self, random_move_board_list: Tuple[List[chess.Move], List[LczeroBoard]], tiny_lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            board_tensor = board.to_config_tensor()
            uci_moves = [move.uci() for move in move_list[:i]]
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = lczero_utils.board_from_backend(tiny_lczero_backend, lczero_game, planes=13)
            assert (board_tensor == lczero_input_tensor[:13]).all()

    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
    def test_board_to_input_tensor(
        self, random_move_board_list: Tuple[List[chess.Move], List[LczeroBoard]], tiny_lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            board_tensor = board.to_input_tensor()
            uci_moves = [move.uci() for move in move_list[:i]]
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = lczero_utils.board_from_backend(tiny_lczero_backend, lczero_game)
            # assert (board_tensor == lczero_input_tensor).all()
            for plane in range(112):
                assert (board_tensor[plane] == lczero_input_tensor[plane]).all()


class TestRepetition:
    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
    def test_board_to_config_tensor(
        self, repetition_move_board_list: Tuple[List[chess.Move], List[LczeroBoard]], tiny_lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            uci_moves = [move.uci() for move in move_list[:i]]
            board_tensor = board.to_config_tensor()
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = lczero_utils.board_from_backend(tiny_lczero_backend, lczero_game, planes=13)
            assert (board_tensor == lczero_input_tensor[:13]).all()

    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
    def test_board_to_input_tensor(
        self, repetition_move_board_list: Tuple[List[chess.Move], List[LczeroBoard]], tiny_lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            uci_moves = [move.uci() for move in move_list[:i]]
            board_tensor = board.to_input_tensor()
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = lczero_utils.board_from_backend(tiny_lczero_backend, lczero_game)
            assert (board_tensor == lczero_input_tensor).all()


class TestLong:
    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
    def test_board_to_config_tensor(
        self, long_move_board_list: Tuple[List[chess.Move], List[LczeroBoard]], tiny_lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            uci_moves = [move.uci() for move in move_list[:i]]
            board_tensor = board.to_config_tensor()
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = lczero_utils.board_from_backend(tiny_lczero_backend, lczero_game, planes=13)
            assert (board_tensor == lczero_input_tensor[:13]).all()

    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
    def test_board_to_input_tensor(
        self, long_move_board_list: Tuple[List[chess.Move], List[LczeroBoard]], tiny_lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            uci_moves = [move.uci() for move in move_list[:i]]
            board_tensor = board.to_input_tensor()
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = lczero_utils.board_from_backend(tiny_lczero_backend, lczero_game)
            assert (board_tensor == lczero_input_tensor).all()


class TestStability:
    def test_encode_decode(self, random_move_board_list: Tuple[List[chess.Move], List[LczeroBoard]]):
        """
        Test that encoding and decoding a move is the identity.
        """
        us, them = chess.WHITE, chess.BLACK
        for move, board in zip(*random_move_board_list):
            encoded_move = LczeroBoard.encode_move(move, us)
            decoded_move = board.decode_move(encoded_move)
            assert move == decoded_move
            us, them = them, us


class TestBackend:
    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
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
            policy_indices = [LczeroBoard.encode_move(move, board.turn) for move in board.legal_moves]
            assert len(lczero_policy_indices) == len(policy_indices)
            assert set(lczero_policy_indices) == set(policy_indices)

    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
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
            policy_indices = [LczeroBoard.encode_move(move, board.turn) for move in board.legal_moves]
            assert len(lczero_policy_indices) == len(policy_indices)
            assert set(lczero_policy_indices) == set(policy_indices)
