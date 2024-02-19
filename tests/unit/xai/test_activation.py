"""Activation lens tests.
"""

from lczerolens.xai import ActivationLens


class TestLens:
    def test_is_compatible(self, tiny_wrapper):
        lens = ActivationLens()
        assert lens.is_compatible(tiny_wrapper)
