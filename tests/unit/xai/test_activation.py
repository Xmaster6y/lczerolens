"""Activation lens tests."""

from lczerolens import Lens
from lczerolens.xai import ActivationLens


class TestLens:
    def test_is_compatible(self, tiny_wrapper):
        lens = Lens.from_name("activation")
        assert isinstance(lens, ActivationLens)
        assert lens.is_compatible(tiny_wrapper)
