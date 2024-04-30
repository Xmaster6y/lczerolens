"""Wrapper tests."""

import chess

from lczerolens.game import WrapperSampler


class TestWrapperSampler:
    def test_get_utility_tiny(self, tiny_wrapper):
        """Test get_utility method."""
        board = chess.Board()
        sampler = WrapperSampler(wrapper=tiny_wrapper)
        utility, _, _ = sampler.get_utility(board)
        assert utility.shape[0] == 20

    def test_get_utility_winner(self, winner_wrapper):
        """Test get_utility method."""
        board = chess.Board()
        sampler = WrapperSampler(wrapper=winner_wrapper)
        utility, _, _ = sampler.get_utility(board)
        assert utility.shape[0] == 20
