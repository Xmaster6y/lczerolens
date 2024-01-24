"""
Wrapper tests.
"""

import chess
import torch
from lczero.backends import GameState

from lczerolens.utils import lczero as lczero_utils
from lczerolens.xai import LrpWrapper


class TestWrapper:
    def test_load_wrapper(self, ensure_network):
        """
        Test that the wrapper loads.
        """
        wrapper = LrpWrapper.from_path("assets/tinygyal-8.onnx")
        assert wrapper.model is not None

    def test_wrapper_prediction(self, lczero_backend, ensure_network):
        """
        Test that the wrapper prediction works.
        """
        wrapper = LrpWrapper.from_path("assets/tinygyal-8.onnx")
        board = chess.Board()
        out = wrapper.predict(board)
        policy = out["policy"]
        value = out["value"]
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
            lczero_backend, lczero_game
        )
        assert torch.allclose(policy, lczero_policy, atol=1e-4)
        assert torch.allclose(value, lczero_value)
