"""CRP lens tests."""

from lczerolens import LensFactory
from lczerolens.lenses import CrpLens


class TestLens:
    def test_is_compatible(self, tiny_model):
        lens = LensFactory.from_name("crp")
        assert isinstance(lens, CrpLens)
        assert lens.is_compatible(tiny_model)
