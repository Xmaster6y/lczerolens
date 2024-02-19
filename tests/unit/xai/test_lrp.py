"""LRP lens tests.
"""

import chess
import torch

from lczerolens.xai import LrpLens


class TestLens:
    def test_is_compatible(self, tiny_wrapper):
        lens = LrpLens()
        assert lens.is_compatible(tiny_wrapper)

    def test_empty_board(self, tiny_wrapper):
        """
        Test that the wrapper prediction works.
        """
        lens = LrpLens()
        board = chess.Board(fen=None)
        out = lens.analyse_board(board, tiny_wrapper)
        assert torch.allclose(out.abs()[:104].sum(0).view(64), torch.zeros(64))
