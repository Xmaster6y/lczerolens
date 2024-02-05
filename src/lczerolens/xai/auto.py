"""
Auto lens module
"""

from .attention import AttentionLens
from .crp import CrpLens
from .lens import Lens
from .lrp import LrpLens


class AutoLens:
    """
    Auto lens constructor.
    """

    lens_types = ["attention", "lrp", "crp"]

    @staticmethod
    def from_type(lens_type: str) -> Lens:
        """
        Create a lens from the given type.
        """
        if lens_type == "attention":
            return AttentionLens()
        if lens_type == "lrp":
            return LrpLens()
        if lens_type == "crp":
            return CrpLens()
        raise ValueError(f"Unknown lens type: {lens_type}")

    @classmethod
    def from_wrapper(cls, wrapper) -> Lens:
        """
        Create a lens from the given wrapper.
        """
        for lens_type in cls.lens_types:
            lens = cls.from_type(lens_type)
            if lens.is_compatible(wrapper):
                return lens
        raise ValueError(f"No compatible lens found for wrapper: {wrapper}")
