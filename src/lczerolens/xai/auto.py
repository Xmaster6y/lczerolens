"""
Auto lens module
"""

from .lens import Lens
from .lenses.attention import AttentionLens
from .lenses.crp import CrpLens
from .lenses.lrp import LrpLens
from .lenses.policy import PolicyLens


class AutoLens:
    """
    Auto lens constructor.
    """

    lens_types = ["attention", "lrp", "crp", "policy"]

    @staticmethod
    def from_type(lens_type: str) -> Lens:
        """
        Create a lens from the given type.
        """
        if lens_type == "attention":
            return AttentionLens()
        elif lens_type == "lrp":
            return LrpLens()
        elif lens_type == "crp":
            return CrpLens()
        elif lens_type == "policy":
            return PolicyLens()
        else:
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
