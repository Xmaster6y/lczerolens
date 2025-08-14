"""
Tests for the board.
"""

from typing import List, Tuple
import pytest
import chess
from lczero.backends import GameState

from lczerolens import backends as lczero_utils
from lczerolens.board import LczeroBoard, InputEncoding


class TestInputEncoding:
    @pytest.mark.parametrize(
        "input_encoding_expected",
        [
            (InputEncoding.INPUT_CLASSICAL_112_PLANE, 32),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED, 32 * 8),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY_REPEATED, 32 * 8),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY_ZEROS, 32),
        ],
    )
    def test_sum_initial_config_planes(self, input_encoding_expected):
        """
        Test the sum of the config planes for the initial board.
        """
        input_encoding, expected = input_encoding_expected
        board = LczeroBoard()
        board_tensor = board.to_input_tensor(input_encoding=input_encoding)
        assert board_tensor[:104].sum() == expected

    @pytest.mark.parametrize(
        "input_encoding_expected",
        [
            (InputEncoding.INPUT_CLASSICAL_112_PLANE, 31 + 32 * 4),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED, 31 + 32 * 7),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY_REPEATED, 31 * 8),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY_ZEROS, 31),
        ],
    )
    def test_sum_qga_config_planes(self, input_encoding_expected):
        """
        Test the sum of the config planes for the queen's gambit accepted.
        """
        input_encoding, expected = input_encoding_expected
        board = LczeroBoard()
        moves = [
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("d7d5"),
            chess.Move.from_uci("c2c4"),
            chess.Move.from_uci("d5c4"),
        ]
        for move in moves:
            board.push(move)
        board_tensor = board.to_input_tensor(input_encoding=input_encoding)
        assert board_tensor[:104].sum() == expected


@pytest.mark.backends
class TestWithBackend:
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


@pytest.mark.backends
class TestRepetition:
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


@pytest.mark.backends
class TestLong:
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


@pytest.mark.backends
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
            policy_indices = [LczeroBoard.encode_move(move, board.turn) for move in board.legal_moves]
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
            policy_indices = [LczeroBoard.encode_move(move, board.turn) for move in board.legal_moves]
            assert len(lczero_policy_indices) == len(policy_indices)
            assert set(lczero_policy_indices) == set(policy_indices)
