"""
Wrapper tests.
"""

from lczerolens.xai import CrpLens


class TestWrapper:
    def test_load_wrapper(self, tiny_wrapper):
        """
        Test that the wrapper loads.
        """
        lens = CrpLens()
        assert lens.is_compatible(tiny_wrapper)
