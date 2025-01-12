"""LRP lens tests."""

from lczerolens import Lens
from lczerolens.lenses import LrpLens


class TestLens:
    def test_is_compatible(self, tiny_model):
        lens = Lens.from_name("lrp")
        assert isinstance(lens, LrpLens)
        assert lens.is_compatible(tiny_model)
