"""
Wrapper tests.
"""

import chess
import torch
from lczero.backends import GameState

from lczerolens import board_utils
from lczerolens.utils import lczero as lczero_utils


class TestSeNet:
    def test_senet_prediction(self, lczero_backend, tiny_senet):
        """
        Test that the wrapper prediction works.
        """

        board = chess.Board()
        out = tiny_senet(
            board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
        )
        policy = out["policy"]
        value = out["value"]
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
            lczero_backend, lczero_game
        )
        assert torch.allclose(policy, lczero_policy, atol=1e-4)
        assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_senet_prediction_random(
        self, lczero_backend, tiny_senet, random_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_senet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                lczero_backend, lczero_game
            )
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_senet_prediction_repetition(
        self, lczero_backend, tiny_senet, repetition_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_senet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                lczero_backend, lczero_game
            )
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_senet_prediction_long(
        self, lczero_backend, tiny_senet, long_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_senet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                lczero_backend, lczero_game
            )
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)
