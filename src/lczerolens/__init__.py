"""Main module for the lczerolens package."""

from importlib.metadata import PackageNotFoundError, version

from .board import LczeroBoard
from .model import LczeroModel

try:
    __version__ = version("lczerolens")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "LczeroBoard",
    "LczeroModel",
]
