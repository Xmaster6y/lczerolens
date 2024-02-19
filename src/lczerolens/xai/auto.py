"""Auto lens module
"""

from .lens import Lens
from .lenses.activation import ActivationLens
from .lenses.crp import CrpLens
from .lenses.lrp import LrpLens
from .lenses.policy import PolicyLens
from .lenses.probing import ProbingLens


class AutoLens:
    """
    Auto lens constructor.
    """

    lens_types = ["activation", "lrp", "crp", "policy", "probing"]
    exclude_from_wrapper = ["probing"]

    @staticmethod
    def from_type(lens_type: str, **kwargs) -> Lens:
        """
        Create a lens from the given type.
        """
        if lens_type == "activation":
            return ActivationLens()
        elif lens_type == "lrp":
            return LrpLens()
        elif lens_type == "crp":
            return CrpLens()
        elif lens_type == "policy":
            return PolicyLens()
        elif lens_type == "probing":
            return ProbingLens(**kwargs)
        else:
            raise ValueError(f"Unknown lens type: {lens_type}")

    @classmethod
    def from_wrapper(cls, wrapper, **kwargs) -> Lens:
        """
        Create a lens from the given wrapper.
        """
        for lens_type in cls.lens_types:
            if lens_type in cls.exclude_from_wrapper:
                continue
            lens = cls.from_type(lens_type)
            if lens.is_compatible(wrapper):
                return lens
        raise ValueError(f"No compatible lens found for wrapper: {wrapper}")
