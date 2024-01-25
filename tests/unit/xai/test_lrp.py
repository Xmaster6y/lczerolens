"""
Wrapper tests.
"""

from lczerolens.adapt import ModelWrapper
from lczerolens.xai import LrpLens


class TestWrapper:
    def test_load_wrapper(self, ensure_network):
        """
        Test that the wrapper loads.
        """
        wrapper = ModelWrapper.from_path("assets/tinygyal-8.onnx")
        lens = LrpLens()
        assert lens.is_compatible(wrapper)
