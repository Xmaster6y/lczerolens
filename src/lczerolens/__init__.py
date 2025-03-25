"""Main module for the lczerolens package."""

from importlib.metadata import PackageNotFoundError, version

from .board import LczeroBoard, InputEncoding
from .model import LczeroModel, Flow
from .lens import Lens
from . import lenses, concepts, play

try:
    __version__ = version("lczerolens")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "LczeroBoard",
    "LczeroModel",
    "Flow",
    "InputEncoding",
    "Lens",
    "lenses",
    "concepts",
    "play",
]
