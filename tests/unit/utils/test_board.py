"""
Tests for the board utils.
"""

import torch
from lczero.backends import Backend, GameState

from lczerolens import board_utils


def board_from_backend(
    lczero_backend: Backend, lczero_game: GameState, planes: int = 112
):
    """
    Create a board from the lczero backend.
    """
    lczero_input = lczero_game.as_input(lczero_backend)
    lczero_input_tensor = torch.zeros((112, 64), dtype=torch.float)
    for plane in range(planes):
        mask_str = f"{lczero_input.mask(plane):b}".zfill(64)
        lczero_input_tensor[plane] = torch.tensor(
            tuple(map(int, reversed(mask_str))), dtype=torch.float
        ) * lczero_input.val(plane)
    return lczero_input_tensor.view((112, 8, 8))


class TestWithBackend:
    def test_board_to_tensor13x8x8(
        self, random_move_board_list, lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            board_tensor = board_utils.board_to_tensor13x8x8(board)
            uci_moves = [move.uci() for move in move_list[:i]]
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = board_from_backend(
                lczero_backend, lczero_game, planes=13
            )
            assert (board_tensor == lczero_input_tensor[:13]).all()

    def test_board_to_tensor112x8x8(
        self, random_move_board_list, lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            board_tensor = board_utils.board_to_tensor112x8x8(board)
            uci_moves = [move.uci() for move in move_list[:i]]
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = board_from_backend(
                lczero_backend, lczero_game
            )
            # assert (board_tensor == lczero_input_tensor).all()
            for plane in range(112):
                assert (
                    board_tensor[plane] == lczero_input_tensor[plane]
                ).all()


class TestRepetition:
    def test_board_to_tensor13x8x8(
        self, repetition_move_board_list, lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            uci_moves = [move.uci() for move in move_list[:i]]
            fens = [board.fen().split(" ")[0] for board in board_list[:i]]
            board_tensor = board_utils.board_to_tensor13x8x8(board, fens=fens)
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = board_from_backend(
                lczero_backend, lczero_game, planes=13
            )
            assert (board_tensor == lczero_input_tensor[:13]).all()

    def test_board_to_tensor112x8x8(
        self, repetition_move_board_list, lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            uci_moves = [move.uci() for move in move_list[:i]]
            fens = [board.fen().split(" ")[0] for board in board_list[:i]]
            board_tensor = board_utils.board_to_tensor112x8x8(board, fens=fens)
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = board_from_backend(
                lczero_backend, lczero_game
            )
            assert (board_tensor == lczero_input_tensor).all()


class TestLong:
    def test_board_to_tensor13x8x8(self, long_move_board_list, lczero_backend):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            uci_moves = [move.uci() for move in move_list[:i]]
            fens = [board.fen().split(" ")[0] for board in board_list[:i]]
            board_tensor = board_utils.board_to_tensor13x8x8(board, fens=fens)
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = board_from_backend(
                lczero_backend, lczero_game, planes=13
            )
            assert (board_tensor == lczero_input_tensor[:13]).all()

    def test_board_to_tensor112x8x8(
        self, long_move_board_list, lczero_backend
    ):
        """
        Test that the board to tensor function works.
        """
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            uci_moves = [move.uci() for move in move_list[:i]]
            fens = [board.fen().split(" ")[0] for board in board_list[:i]]
            board_tensor = board_utils.board_to_tensor112x8x8(board, fens=fens)
            lczero_game = GameState(moves=uci_moves)
            lczero_input_tensor = board_from_backend(
                lczero_backend, lczero_game
            )
            assert (board_tensor == lczero_input_tensor).all()
