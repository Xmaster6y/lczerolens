"""Unified reference and official-Lczero search interfaces."""

from .capabilities import SearchAdapterCapability, SearchAdapterCapabilityError
from .limits import Depth, Nodes, SearchLimit, Simulations, Time, Visits
from .lczero import LczeroSearch
from .reference import ReferenceSearch
from .result import SearchAction, SearchEvidenceUnavailable, SearchResult, SearchRoot
from .trace import SearchTrace

__all__ = [
    "Depth",
    "LczeroSearch",
    "Nodes",
    "ReferenceSearch",
    "SearchAction",
    "SearchAdapterCapability",
    "SearchAdapterCapabilityError",
    "SearchEvidenceUnavailable",
    "SearchLimit",
    "SearchResult",
    "SearchRoot",
    "SearchTrace",
    "Simulations",
    "Time",
    "Visits",
]
