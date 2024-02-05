"""
Wrapper tests.
"""

import chess
import torch
from lczero.backends import GameState

from lczerolens.adapt import MlhFlow, PolicyFlow, ValueFlow, WdlFlow
from lczerolens.utils import lczero as lczero_utils


class TestWrapper:
    def test_load_wrapper(self, tiny_wrapper):
        """
        Test that the wrapper loads.
        """
        assert tiny_wrapper.model is not None

    def test_wrapper_prediction(self, tiny_lczero_backend, tiny_wrapper):
        """
        Test that the wrapper prediction works.
        """
        board = chess.Board()
        out = tiny_wrapper.predict(board)
        policy = out["policy"]
        value = out["value"]
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
            tiny_lczero_backend, lczero_game
        )
        assert torch.allclose(policy, lczero_policy, atol=1e-4)
        assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_random(
        self, tiny_lczero_backend, tiny_wrapper, random_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                tiny_lczero_backend, lczero_game
            )
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_repetition(
        self, tiny_lczero_backend, tiny_wrapper, repetition_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                tiny_lczero_backend, lczero_game
            )
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_wrapper_prediction_long(
        self, tiny_lczero_backend, tiny_wrapper, long_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_wrapper.predict(board)
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(
                moves=[move.uci() for move in move_list[:i]]
            )
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(
                tiny_lczero_backend, lczero_game
            )
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)


class TestFlows:
    def test_policy_flow(self, tiny_wrapper):
        """
        Test that the policy flow works.
        """
        policy_flow = PolicyFlow(tiny_wrapper.model)
        board = chess.Board()
        policy = policy_flow.predict(board)
        wrapper_policy = tiny_wrapper.predict(board)["policy"]
        assert torch.allclose(policy, wrapper_policy)

    def test_value_flow(self, tiny_wrapper):
        """
        Test that the value flow works.
        """
        value_flow = ValueFlow(tiny_wrapper.model)
        board = chess.Board()
        value = value_flow.predict(board)
        wrapper_value = tiny_wrapper.predict(board)["value"]
        assert torch.allclose(value, wrapper_value)

    def test_wdl_flow(self, winner_wrapper):
        """
        Test that the wdl flow works.
        """
        wdl_flow = WdlFlow(winner_wrapper.model)
        board = chess.Board()
        wdl = wdl_flow.predict(board)
        wrapper_wdl = winner_wrapper.predict(board)["wdl"]
        assert torch.allclose(wdl, wrapper_wdl)

    def test_mlh_flow(self, winner_wrapper):
        """
        Test that the mlh flow works.
        """
        mlh_flow = MlhFlow(winner_wrapper.model)
        board = chess.Board()
        mlh = mlh_flow.predict(board)
        wrapper_mlh = winner_wrapper.predict(board)["mlh"]
        assert torch.allclose(mlh, wrapper_mlh)
