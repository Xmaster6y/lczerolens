"""Model tests."""

import chess
import pytest
import torch
from lczero.backends import GameState

from lczerolens import FlowFactory
from lczerolens.encodings import backends as lczero_utils


class TestModel:
    def test_load_model(self, tiny_model):
        """Test that the model loads."""
        tiny_model

    def test_model_prediction(self, tiny_lczero_backend, tiny_model):
        """Test that the model prediction works."""
        board = chess.Board()
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


class TestFlows:
    def test_policy_flow(self, tiny_model):
        """Test that the policy flow works."""
        policy_flow = FlowFactory.from_model("policy", tiny_model)
        board = chess.Board()
        (policy,) = policy_flow(board)
        model_policy = tiny_model(board)[0]["policy"]
        assert torch.allclose(policy, model_policy)

    def test_value_flow(self, tiny_model):
        """Test that the value flow works."""
        value_flow = FlowFactory.from_model("value", tiny_model)
        board = chess.Board()
        (value,) = value_flow(board)
        model_value = tiny_model(board)[0]["value"]
        assert torch.allclose(value, model_value)

    def test_wdl_flow(self, winner_model):
        """Test that the wdl flow works."""
        wdl_flow = FlowFactory.from_model("wdl", winner_model)
        board = chess.Board()
        (wdl,) = wdl_flow(board)
        model_wdl = winner_model(board)[0]["wdl"]
        assert torch.allclose(wdl, model_wdl)

    def test_mlh_flow(self, winner_model):
        """Test that the mlh flow works."""
        mlh_flow = FlowFactory.from_model("mlh", winner_model)
        board = chess.Board()
        (mlh,) = mlh_flow(board)
        model_mlh = winner_model(board)[0]["mlh"]
        assert torch.allclose(mlh, model_mlh)

    def test_incompatible_flows(self, tiny_model, winner_model):
        """Test that the flows raise an error *
        when the model is incompatible.
        """
        with pytest.raises(ValueError):
            FlowFactory.from_model("value", winner_model)
        with pytest.raises(ValueError):
            FlowFactory.from_model("wdl", tiny_model)
        with pytest.raises(ValueError):
            FlowFactory.from_model("mlh", tiny_model)
        with pytest.raises(ValueError):
            FlowFactory.get_subclass("value")(tiny_model)
