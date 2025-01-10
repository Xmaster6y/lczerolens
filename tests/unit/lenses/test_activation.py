"""Activation lens tests."""

from lczerolens import Lens
from lczerolens.lenses import ActivationLens


class TestLens:
    def test_is_compatible(self, tiny_model):
        lens = Lens.from_name("activation")
        assert isinstance(lens, ActivationLens)
        assert lens.is_compatible(tiny_model)
