"""Activation lens tests."""

from lczerolens import Lens
from lczerolens.lenses import ActivationLens
from lczerolens.board import LczeroBoard


class TestLens:
    def test_is_compatible(self, tiny_model):
        lens = Lens.from_name("activation")
        assert isinstance(lens, ActivationLens)
        assert lens.is_compatible(tiny_model)

    def test_analyse_board(self, tiny_model):
        lens = ActivationLens(pattern=r".*")
        board = LczeroBoard()
        results = lens.analyse(tiny_model, board)

        assert len(results) > 0
        for key in results:
            assert key.endswith("_output")

    def test_analyse_with_inputs(self, tiny_model):
        lens = ActivationLens(pattern=r".*")
        board = LczeroBoard()
        results = lens.analyse(tiny_model, board, save_inputs=True)

        input_keys = [k for k in results if k.endswith("_input")]
        output_keys = [k for k in results if k.endswith("_output")]

        assert len(input_keys) > 0
        assert len(output_keys) > 0

    def test_analyse_specific_modules(self, tiny_model):
        lens = ActivationLens(pattern=r".*conv.*")
        board = LczeroBoard()
        results = lens.analyse(tiny_model, board)

        assert len(results) > 0
        for key in results:
            module_name = key.replace("_output", "")
            assert "conv" in module_name
