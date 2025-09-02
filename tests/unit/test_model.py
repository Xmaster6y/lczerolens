"""Model tests."""

import pytest
import torch
from lczero.backends import GameState

from lczerolens.model import LczeroBoard, PolicyFlow, ValueFlow, WdlFlow, MlhFlow, ForceValue, LczeroModel
from lczerolens import backends as lczero_utils


@pytest.mark.backends
class TestModel:
    def test_model_prediction(self, tiny_lczero_backend, tiny_model):
        """Test that the model prediction works."""
        board = LczeroBoard()
        (out,) = tiny_model(board)
        policy = out["policy"]
        value = out["value"]
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
        assert torch.allclose(policy, lczero_policy, atol=1e-4)
        assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_model_prediction_random(self, tiny_lczero_backend, tiny_model, random_move_board_list):
        """Test that the model prediction works."""
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_model(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_model_prediction_repetition(self, tiny_lczero_backend, tiny_model, repetition_move_board_list):
        """Test that the model prediction works."""
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_model(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_model_prediction_long(self, tiny_lczero_backend, tiny_model, long_move_board_list):
        """Test that the model prediction works."""
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_model(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)


class TestManageModels:
    def test_model_from_hf(self):
        """Test that the model save and load works."""
        board = LczeroBoard()
        model = LczeroModel.from_hf("lczerolens/maia-1100")
        output = model(board)
        assert "policy" in output
        assert "wdl" in output


class TestFlows:
    def test_policy_flow(self, tiny_model):
        """Test that the policy flow works."""
        policy_flow = PolicyFlow.from_model(tiny_model.module)
        board = LczeroBoard()
        output = policy_flow(board)
        assert "value" not in output

    def test_value_flow(self, tiny_model):
        """Test that the value flow works."""
        value_flow = ValueFlow.from_model(tiny_model.module)
        board = LczeroBoard()
        output = value_flow(board)
        assert "policy" not in output

    def test_wdl_flow(self, winner_model):
        """Test that the wdl flow works."""
        wdl_flow = WdlFlow.from_model(winner_model.module)
        board = LczeroBoard()
        output = wdl_flow(board)
        assert "policy" not in output

    def test_mlh_flow(self, winner_model):
        """Test that the mlh flow works."""
        mlh_flow = MlhFlow.from_model(winner_model.module)
        board = LczeroBoard()
        output = mlh_flow(board)
        assert "policy" not in output

    def test_force_value(self, tiny_model):
        """Test that the force value flow works."""
        force_value = ForceValue.from_model(tiny_model.module)
        board = LczeroBoard()
        output = force_value(board)
        assert "value" in output

    def test_force_value_wdl(self, winner_model):
        """Test that the force value flow works."""
        force_value = ForceValue.from_model(winner_model.module)
        board = LczeroBoard()
        output = force_value(board)
        assert "value" in output

        value = output["wdl"] @ torch.tensor([1.0, 0.0, -1.0], device=output.device)
        assert torch.allclose(output["value"], value)

    def test_incompatible_flows(self, tiny_model, winner_model):
        """Test that the flows raise an error *
        when the model is incompatible.
        """
        with pytest.raises(ValueError):
            ValueFlow.from_model(winner_model)
        with pytest.raises(ValueError):
            WdlFlow.from_model(tiny_model)
        with pytest.raises(ValueError):
            MlhFlow.from_model(tiny_model)
