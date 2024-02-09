"""
Wrapper tests.
"""

from lczerolens.xai import AttentionLens


class TestWrapper:
    def test_load_wrapper(self, tiny_wrapper):
        """
        Test that the wrapper loads.
        """
        lens = AttentionLens()
        assert lens.is_compatible(tiny_wrapper) is False
