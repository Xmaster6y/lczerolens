"""Capabilities of search producers before a search is started."""

from __future__ import annotations

from enum import Enum


class SearchAdapterCapability(str, Enum):
    """One optional input capability implemented by a search adapter."""

    ROOT_EVALUATION_REPLACEMENT = "root_evaluation_replacement"


class SearchAdapterCapabilityError(RuntimeError):
    """Raised when a search adapter cannot apply a requested input."""


def require_adapter_capability(
    capabilities: frozenset[SearchAdapterCapability], capability: SearchAdapterCapability
) -> None:
    """Fail closed unless ``capability`` is explicitly advertised."""
    if not isinstance(capability, SearchAdapterCapability):
        raise TypeError("Search adapter capability checks require a SearchAdapterCapability value.")
    if capability not in capabilities:
        raise SearchAdapterCapabilityError(f"Search adapter does not support {capability.value}.")


__all__ = ["SearchAdapterCapability", "SearchAdapterCapabilityError", "require_adapter_capability"]
