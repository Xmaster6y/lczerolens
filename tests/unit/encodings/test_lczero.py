"""
LCZero utils tests.
"""

import torch
from lczero.backends import GameState

from lczerolens.encodings import backends as lczero_utils


class TestExecution:
    def test_describenet(self):
        """
        Test that the describenet function works.
        """
        description = lczero_utils.describenet("assets/tinygyal-8.pb.gz")
        assert isinstance(description, str)
        assert "Minimal Lc0 version:" in description

    def test_convertnet(self):
        """
        Test that the convert_to_onnx function works.
        """
        conversion = lczero_utils.convert_to_onnx("assets/tinygyal-8.pb.gz", "assets/tinygyal-8.onnx")
        assert isinstance(conversion, str)
        assert "INPUT_CLASSICAL_112_PLANE" in conversion

    def test_generic_command(self):
        """
        Test that the generic command function works.
        """
        generic_command = lczero_utils.generic_command(["--help"])
        assert isinstance(generic_command, str)
        assert "Usage: lc0" in generic_command

    def test_board_from_backend(self, tiny_lczero_backend):
        """
        Test that the board from backend function works.
        """
        lczero_game = GameState()
        lczero_board_tensor = lczero_utils.board_from_backend(tiny_lczero_backend, lczero_game)
        assert lczero_board_tensor.shape == (112, 8, 8)

    def test_prediction_from_backend(self, tiny_lczero_backend):
        """
        Test that the prediction from backend function works.
        """
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
        assert lczero_policy.shape == (1858,)
        assert (lczero_value >= -1) and (lczero_value <= 1)
        lczero_policy_softmax, _ = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game, softmax=True)
        assert lczero_policy_softmax.shape == (1858,)
        assert (lczero_policy_softmax >= 0).all() and (lczero_policy_softmax <= 1).all()
        assert torch.softmax(lczero_policy, dim=0).allclose(lczero_policy_softmax, atol=1e-4)
