"""Typed, producer-independent search limits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

from .trace import SearchBudgetUnit


@dataclass(frozen=True)
class _CountLimit:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 1:
            raise ValueError(f"{type(self).__name__} requires a positive integer.")


@dataclass(frozen=True)
class Nodes(_CountLimit):
    """Maximum engine node count."""

    unit = SearchBudgetUnit.NODES


@dataclass(frozen=True)
class Visits(_CountLimit):
    """Maximum root visit count."""

    unit = SearchBudgetUnit.VISITS


@dataclass(frozen=True)
class Simulations(_CountLimit):
    """Exact number of reference-search simulations."""

    unit = SearchBudgetUnit.SIMULATIONS


@dataclass(frozen=True)
class Depth(_CountLimit):
    """Maximum search depth in plies."""

    unit = SearchBudgetUnit.DEPTH


@dataclass(frozen=True)
class Time:
    """Wall-clock limit in milliseconds."""

    milliseconds: float
    unit = SearchBudgetUnit.TIME_MS

    def __post_init__(self) -> None:
        if (
            isinstance(self.milliseconds, bool)
            or not isinstance(self.milliseconds, int | float)
            or not math.isfinite(self.milliseconds)
            or self.milliseconds <= 0
        ):
            raise ValueError("Time requires finite, positive milliseconds.")

    @property
    def value(self) -> float:
        return self.milliseconds


SearchLimit: TypeAlias = Nodes | Visits | Simulations | Time | Depth


__all__ = ["Depth", "Nodes", "SearchLimit", "Simulations", "Time", "Visits"]
