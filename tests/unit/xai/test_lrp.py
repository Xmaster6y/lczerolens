"""LRP lens tests."""

import chess
import torch

from lczerolens import LensFactory
from lczerolens.lenses import LrpLens


class TestLens:
    def test_is_compatible(self, tiny_model):
        lens = LensFactory.from_name("lrp")
        assert isinstance(lens, LrpLens)
        assert lens.is_compatible(tiny_model)

    def test_empty_board(self, tiny_model):
        """
        Test that the wrapper prediction works.
        """
        lens = LrpLens()
        board = chess.Board(fen=None)
        (rel,) = lens.analyse(board, tiny_model)
        assert torch.allclose(rel.abs()[0, :104].sum(0).view(64), torch.zeros(64))
