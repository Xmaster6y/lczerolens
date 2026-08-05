"""Natural immutable result shared by every search producer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import chess

from .trace import (
    PositionEvaluation,
    PrincipalVariation,
    RootAction,
    SearchCapability,
    SearchTrace,
    ValuePerspective,
)


class SearchEvidenceUnavailable(RuntimeError):
    """Raised when a result source did not expose requested search evidence."""


@dataclass(frozen=True)
class SearchAction:
    """One root action with exactly the fields exposed by its producer."""

    move: str
    prior: float | None
    visits: int | None
    total_value: float | None
    mean_value: float | None
    exploration: float | None
    perspective: ValuePerspective
    evaluation: PositionEvaluation | None
    leaf_evaluation: PositionEvaluation | None
    principal_variation: PrincipalVariation | None

    @classmethod
    def from_root_action(cls, action: RootAction) -> "SearchAction":
        statistics = action.statistics
        return cls(
            statistics.move,
            statistics.prior,
            statistics.visits,
            statistics.total_value,
            statistics.mean_value,
            statistics.exploration,
            statistics.perspective,
            action.evaluation,
            action.leaf_evaluation,
            action.principal_variation,
        )


class SearchRoot(Mapping[str, SearchAction]):
    """Immutable move-keyed root actions in canonical UCI order."""

    def __init__(self, actions: Sequence[SearchAction]):
        self._actions = tuple(actions)
        self._by_move = {action.move: action for action in self._actions}
        if len(self._actions) != len(self._by_move):
            raise ValueError("Search root actions must have unique moves.")
        if tuple(self._by_move) != tuple(sorted(self._by_move)):
            raise ValueError("Search root actions must use canonical UCI order.")

    def __getitem__(self, move: str | chess.Move) -> SearchAction:
        key = move if isinstance(move, str) else move.uci()
        return self._by_move[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_move)

    def __len__(self) -> int:
        return len(self._actions)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SearchRoot) and self._actions == other._actions


@dataclass(frozen=True)
class SearchResult:
    """Final search decision plus its immutable producer evidence."""

    move: chess.Move
    evaluation: PositionEvaluation | None
    principal_variation: PrincipalVariation | None
    trace: SearchTrace
    _root: SearchRoot | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        board = chess.Board(self.trace.root_fen)
        if self.move not in board.legal_moves:
            raise ValueError("Search result move must be legal in the trace root position.")
        selection = self.trace.snapshots[-1].selection
        if selection is None or selection.move != self.move.uci():
            raise ValueError("Search result move must match the final trace selection.")
        if self._root is not None:
            if self.move.uci() not in self._root:
                raise ValueError("An exposed search root must contain the selected move.")
            selected = self._root[self.move]
            if self.principal_variation != selected.principal_variation:
                raise ValueError("Search result principal variation must match the selected root action.")
        if self.principal_variation is not None and self.principal_variation.moves[0] != self.move.uci():
            raise ValueError("Search result principal variation must start with its selected move.")

    @classmethod
    def from_trace(cls, trace: SearchTrace) -> "SearchResult":
        """Construct the natural final view without inventing absent evidence."""
        if not isinstance(trace, SearchTrace):
            raise TypeError("SearchResult.from_trace expects a SearchTrace.")
        final = trace.snapshots[-1]
        if final.selection is None:
            raise SearchEvidenceUnavailable(
                f"Search source {trace.provenance.source!r} did not expose a final selected move."
            )
        root = (
            SearchRoot(
                tuple(
                    SearchAction.from_root_action(action)
                    for action in sorted(final.actions, key=lambda item: item.statistics.move)
                )
            )
            if final.actions is not None
            else None
        )
        selected = root[final.selection.move] if root is not None else None
        return cls(
            chess.Move.from_uci(final.selection.move),
            final.evaluation,
            selected.principal_variation if selected is not None else None,
            trace,
            root,
        )

    @property
    def root(self) -> SearchRoot:
        """Exposed root actions or an actionable absence error."""
        if self._root is None:
            raise SearchEvidenceUnavailable(
                f"Search source {self.trace.provenance.source!r} did not expose root action statistics."
            )
        return self._root

    @property
    def has_root_actions(self) -> bool:
        return self._root is not None

    @property
    def has_snapshots(self) -> bool:
        return self.trace.has_snapshots

    @property
    def has_events(self) -> bool:
        return self.trace.has_events

    @property
    def is_replayable(self) -> bool:
        return self.trace.is_replayable

    @property
    def capability(self) -> SearchCapability:
        """Legacy ordered capability retained for precise existing consumers."""
        return self.trace.capability


__all__ = [
    "SearchAction",
    "SearchEvidenceUnavailable",
    "SearchResult",
    "SearchRoot",
]
