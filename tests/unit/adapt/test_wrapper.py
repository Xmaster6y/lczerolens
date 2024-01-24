"""
Wrapper tests.
"""

import chess
import torch
from lczero.backends import GameState

from lczerolens.adapt import ModelWrapper
from lczerolens.utils import lczero as lczero_utils


class TestWrapper:
    def test_load_wrapper(self, ensure_network):
        """
        Test that the wrapper loads.
        """
        wrapper = ModelWrapper.from_path("assets/tinygyal-8.onnx")
        assert wrapper.model is not None

    def test_wrapper_prediction(self, lczero_backend, ensure_network):
        """
        Test that the wrapper prediction works.
        """
        wrapper = ModelWrapper.from_path("assets/tinygyal-8.onnx")
        board = chess.Board()
        out = wrapper.predict(board)
        policy = out["policy"]
        value = out["value"]
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
            lczero_backend, lczero_game
        )
        argsort = torch.argsort(policy, stable=True)
        lczero_argsort = torch.argsort(lczero_policy, stable=True)
        if (argsort != lczero_argsort).sum() > 10:
            raise ValueError
        assert torch.allclose(policy, lczero_policy, atol=1e-4)
        assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_random(
        self, lczero_backend, ensure_network, random_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """
        wrapper = ModelWrapper.from_path("assets/tinygyal-8.onnx")
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            out = wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                lczero_backend, lczero_game
            )
            argsort = torch.argsort(policy, stable=True)
            lczero_argsort = torch.argsort(lczero_policy, stable=True)
            if (argsort != lczero_argsort).sum() > 10:
                raise ValueError
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_repetition(
        self, lczero_backend, ensure_network, repetition_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """
        wrapper = ModelWrapper.from_path("assets/tinygyal-8.onnx")
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            out = wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                lczero_backend, lczero_game
            )
            argsort = torch.argsort(policy, stable=True)
            lczero_argsort = torch.argsort(lczero_policy, stable=True)
            if (argsort != lczero_argsort).sum() > 10:
                raise ValueError
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_long(
        self, lczero_backend, ensure_network, long_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """
        wrapper = ModelWrapper.from_path("assets/tinygyal-8.onnx")
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            out = wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                lczero_backend, lczero_game
            )
            argsort = torch.argsort(policy, stable=True)
            lczero_argsort = torch.argsort(lczero_policy, stable=True)
            if (argsort != lczero_argsort).sum() > 10:
                raise ValueError
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)
