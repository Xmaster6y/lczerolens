"""
Wrapper tests.
"""

import chess
import torch

from lczerolens.xai import LrpLens


class TestWrapper:
    def test_load_wrapper(self, tiny_wrapper):
        """
        Test that the wrapper loads.
        """
        lens = LrpLens()
        assert lens.is_compatible(tiny_wrapper)

    def test_empty_board(self, tiny_wrapper):
        """
        Test that the wrapper prediction works.
        """
        lens = LrpLens()
        board = chess.Board(fen=None)
        out = lens.compute_heatmap(board, tiny_wrapper)
        assert torch.allclose(out.abs()[:104].sum(0).view(64), torch.zeros(64))
