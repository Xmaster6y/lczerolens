"""Model tests."""

import pytest
import torch
import sys
from lczero.backends import GameState

from lczerolens import Flow, LczeroBoard
from lczerolens import backends as lczero_utils


class TestModel:
    def test_load_model(self, tiny_model):
        """Test that the model loads."""
        tiny_model

    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
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

    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
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

    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
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

    @pytest.mark.skipif(sys.version_info >= (3, 9), reason="lczero.backends is only supported on Python 3.9")
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


class TestFlows:
    def test_policy_flow(self, tiny_model):
        """Test that the policy flow works."""
        policy_flow = Flow.from_model("policy", tiny_model)
        board = LczeroBoard()
        (policy,) = policy_flow(board)
        model_policy = tiny_model(board)["policy"][0]
        assert torch.allclose(policy, model_policy)

    def test_value_flow(self, tiny_model):
        """Test that the value flow works."""
        value_flow = Flow.from_model("value", tiny_model)
        board = LczeroBoard()
        (value,) = value_flow(board)
        model_value = tiny_model(board)["value"][0]
        assert torch.allclose(value, model_value)

    def test_wdl_flow(self, winner_model):
        """Test that the wdl flow works."""
        wdl_flow = Flow.from_model("wdl", winner_model)
        board = LczeroBoard()
        (wdl,) = wdl_flow(board)
        model_wdl = winner_model(board)["wdl"][0]
        assert torch.allclose(wdl, model_wdl)

    def test_mlh_flow(self, winner_model):
        """Test that the mlh flow works."""
        mlh_flow = Flow.from_model("mlh", winner_model)
        board = LczeroBoard()
        (mlh,) = mlh_flow(board)
        model_mlh = winner_model(board)["mlh"][0]
        assert torch.allclose(mlh, model_mlh)

    def test_force_value_flow_value(self, tiny_model):
        """Test that the force value flow works."""
        force_value_flow = Flow.from_model("force_value", tiny_model)
        board = LczeroBoard()
        (value,) = force_value_flow(board)
        model_value = tiny_model(board)["value"][0]
        assert torch.allclose(value, model_value)

    def test_force_value_flow_wdl(self, winner_model):
        """Test that the force value flow works."""
        force_value_flow = Flow.from_model("force_value", winner_model)
        board = LczeroBoard()
        (wdl,) = force_value_flow(board)
        model_wdl = winner_model(board)["wdl"][0]
        model_value = model_wdl @ torch.tensor([1.0, 0.0, -1.0], device=model_wdl.device)
        assert torch.allclose(wdl, model_value)

    def test_incompatible_flows(self, tiny_model, winner_model):
        """Test that the flows raise an error *
        when the model is incompatible.
        """
        with pytest.raises(ValueError):
            Flow.from_model("value", winner_model)
        with pytest.raises(ValueError):
            Flow.from_model("wdl", tiny_model)
        with pytest.raises(ValueError):
            Flow.from_model("mlh", tiny_model)

        with pytest.raises(ValueError):
            Flow._registry["value"](tiny_model)


@Flow.register("test_flow")
class TestFlow(Flow):
    """Test flow."""


class TestFlowRegistry:
    def test_flow_registry_duplicate(self):
        """Test that registering a flow with an existing name raises an error."""
        with pytest.raises(ValueError, match="Flow .* already registered"):

            @Flow.register("test_flow")
            class DuplicateFlow(Flow):
                """Duplicate flow."""

    def test_flow_registry_missing(self, tiny_model):
        """Test that instantiating a non-registered flow raises an error."""
        with pytest.raises(KeyError, match="Flow .* not found"):
            Flow.from_model("non_existent_flow", tiny_model)

    def test_flow_type(self):
        """Test that the flow type is correct."""
        assert TestFlow._flow_type == "test_flow"
