"""Gradient lens tests."""

from lczerolens import Lens
from lczerolens.lenses import GradientLens
from lczerolens.board import LczeroBoard


class TestLens:
    def test_is_compatible(self, tiny_model):
        lens = Lens.from_name("gradient")
        assert isinstance(lens, GradientLens)
        assert lens.is_compatible(tiny_model)

    def test_analyse_board(self, tiny_model):
        lens = GradientLens()
        board = LczeroBoard()
        results = lens.analyse(tiny_model, board)

        assert "input_grad" in results

    def test_analyse_without_input_grad(self, tiny_model):
        lens = GradientLens(input_requires_grad=False)
        board = LczeroBoard()
        results = lens.analyse(tiny_model, board)

        assert "input_grad" not in results

    def test_analyse_specific_modules(self, tiny_model):
        lens = GradientLens(pattern=r".*conv.*relu", input_requires_grad=False)
        board = LczeroBoard()
        results = lens.analyse(tiny_model, board)

        assert len(results) > 0
        for key in results:
            module_name = key.replace("_output_grad", "")
            assert "conv" in module_name
