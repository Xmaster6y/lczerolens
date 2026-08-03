"""Main module for the lczerolens package."""

from importlib.metadata import PackageNotFoundError, version

from .board import LczeroBoard
from .facts import Evidence, EvidenceSet, FactAnalyzer
from .lc0_adapter import Lc0ProcessAdapter, Lc0RootSnapshotParser, Lc0SearchRequest
from .model import LczeroModel
from .reference_search import ReferenceMCTS, replay_root_events

try:
    __version__ = version("lczerolens")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = [
    "LczeroBoard",
    "LczeroModel",
    "Evidence",
    "EvidenceSet",
    "FactAnalyzer",
    "Lc0ProcessAdapter",
    "Lc0RootSnapshotParser",
    "Lc0SearchRequest",
    "ReferenceMCTS",
    "replay_root_events",
]
