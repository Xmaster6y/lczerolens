"""
Wrapper tests.
"""

from lczerolens.adapt import ModelWrapper
from lczerolens.xai import CrpLens


class TestWrapper:
    def test_load_wrapper(self, tiny_ensure_network):
        """
        Test that the wrapper loads.
        """
        wrapper = ModelWrapper.from_path("assets/tinygyal-8.onnx")
        lens = CrpLens()
        assert lens.is_compatible(wrapper)
