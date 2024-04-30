"""CRP lens tests."""

from lczerolens import Lens
from lczerolens.xai import CrpLens


class TestLens:
    def test_is_compatible(self, tiny_wrapper):
        lens = Lens.from_name("crp")
        assert isinstance(lens, CrpLens)
        assert lens.is_compatible(tiny_wrapper)
