"""Wrapper tests."""

import chess
import pytest
import torch
from lczero.backends import GameState

from lczerolens.model import MlhFlow, PolicyFlow, ValueFlow, WdlFlow
from lczerolens.model import lczero as lczero_utils


class TestWrapper:
    def test_load_wrapper(self, tiny_wrapper):
        """Test that the wrapper loads."""
        assert tiny_wrapper.model is not None

    def test_wrapper_prediction(self, tiny_lczero_backend, tiny_wrapper):
        """Test that the wrapper prediction works."""
        board = chess.Board()
        (out,) = tiny_wrapper.predict(board)
        policy = out["policy"]
        value = out["value"]
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
        assert torch.allclose(policy, lczero_policy, atol=1e-4)
        assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_random(self, tiny_lczero_backend, tiny_wrapper, random_move_board_list):
        """Test that the wrapper prediction works."""
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_repetition(self, tiny_lczero_backend, tiny_wrapper, repetition_move_board_list):
        """Test that the wrapper prediction works."""
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_long(self, tiny_lczero_backend, tiny_wrapper, long_move_board_list):
        """Test that the wrapper prediction works."""
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)


class TestFlows:
    def test_policy_flow(self, tiny_wrapper):
        """Test that the policy flow works."""
        policy_flow = PolicyFlow(tiny_wrapper.model)
        board = chess.Board()
        (policy,) = policy_flow.predict(board)
        wrapper_policy = tiny_wrapper.predict(board)[0]["policy"]
        assert torch.allclose(policy, wrapper_policy)

    def test_value_flow(self, tiny_wrapper):
        """Test that the value flow works."""
        value_flow = ValueFlow(tiny_wrapper.model)
        board = chess.Board()
        (value,) = value_flow.predict(board)
        wrapper_value = tiny_wrapper.predict(board)[0]["value"]
        assert torch.allclose(value, wrapper_value)

    def test_wdl_flow(self, winner_wrapper):
        """Test that the wdl flow works."""
        wdl_flow = WdlFlow(winner_wrapper.model)
        board = chess.Board()
        (wdl,) = wdl_flow.predict(board)
        wrapper_wdl = winner_wrapper.predict(board)[0]["wdl"]
        assert torch.allclose(wdl, wrapper_wdl)

    def test_mlh_flow(self, winner_wrapper):
        """Test that the mlh flow works."""
        mlh_flow = MlhFlow(winner_wrapper.model)
        board = chess.Board()
        (mlh,) = mlh_flow.predict(board)
        wrapper_mlh = winner_wrapper.predict(board)[0]["mlh"]
        assert torch.allclose(mlh, wrapper_mlh)

    def test_incompatible_flows(self, tiny_wrapper, winner_wrapper):
        """Test that the flows raise an error *
        when the model is incompatible.
        """
        with pytest.raises(ValueError):
            ValueFlow(winner_wrapper.model)
        with pytest.raises(ValueError):
            WdlFlow(tiny_wrapper.model)
        with pytest.raises(ValueError):
            MlhFlow(tiny_wrapper.model)
