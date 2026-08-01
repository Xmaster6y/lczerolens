"""Main module for the lczerolens package."""

from importlib.metadata import PackageNotFoundError, version

from .board import LczeroBoard
from .model import LczeroModel
from .reference_search import ReferenceMCTS, replay_root_events

try:
    __version__ = version("lczerolens")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "LczeroBoard",
    "LczeroModel",
    "ReferenceMCTS",
    "replay_root_events",
]
