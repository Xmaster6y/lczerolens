"""
Wrapper tests.
"""

import chess
import torch
from lczero.backends import GameState

from lczerolens.game import LczerroModelWrapper
from lczerolens.utils import lczero as lczero_utils


class TestWrapper:
    def test_load_wrapper(self):
        """
        Test that the wrapper loads.
        """
        wrapper = LczerroModelWrapper("assets/tinygyal-8.onnx")
        wrapper.ensure_loaded()
        assert wrapper.model is not None

    def test_wrapper_prediction(self, lczero_backend):
        """
        Test that the wrapper prediction works.
        """
        wrapper = LczerroModelWrapper("assets/tinygyal-8.onnx")
        board = chess.Board()
        policy, _, value = wrapper.prediction(board)
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
            lczero_backend, lczero_game
        )
        assert torch.allclose(policy, lczero_policy, atol=1e-5)
        assert torch.allclose(value, lczero_value)
