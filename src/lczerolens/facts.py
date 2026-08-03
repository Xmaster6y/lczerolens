"""Typed chess facts and exact reference analyzers.

This module describes chess-domain observations without choosing how a consumer
will turn them into labels, tensors, datasets, or metrics.  Every observation
retains the perspective, guarantee, supporting chess objects, and analyzer
provenance needed to interpret it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, TypeAlias, runtime_checkable

import chess


class FactKind(str, Enum):
    """Kinds produced by the bundled reference analyzers."""

    MATERIAL = "material"
    PIECE_PRESENCE = "piece_presence"
    ATTACKS_DEFENDERS = "attacks_defenders"
    CHECK_STATUS = "check_status"
    LEGAL_MOBILITY = "legal_mobility"


class FactScope(str, Enum):
    """Chess object over which a fact is asserted."""

    SIDE = "side"
    PIECE = "piece"
    SQUARE = "square"
    MOVE = "move"
    REGION = "region"
    POSITION = "position"


class FactPerspective(str, Enum):
    """Viewpoint used to resolve relative subjects and values."""

    ABSOLUTE = "absolute"
    WHITE = "white"
    BLACK = "black"
    SIDE_TO_MOVE = "side_to_move"


class Guarantee(str, Enum):
    """Strength and origin of an evidence value."""

    EXACT = "exact"
    HEURISTIC = "heuristic"
    ENGINE_DERIVED = "engine-derived"
    SEARCH_DERIVED = "search-derived"


class HistoryRequirement(str, Enum):
    """Position history needed to establish a fact."""

    NONE = "none"
    LAST_MOVE = "last_move"
    FULL_MOVE_STACK = "full_move_stack"


class UndefinedReason(str, Enum):
    """Why an analyzer could not establish a value."""

    HISTORY_UNAVAILABLE = "history_unavailable"
    INVALID_POSITION = "invalid_position"
    MISSING_KING = "missing_king"


class ChessSide(str, Enum):
    """A concrete side, independent of board perspective."""

    WHITE = "white"
    BLACK = "black"

    @classmethod
    def from_color(cls, color: chess.Color) -> ChessSide:
        """Convert a python-chess color to a concrete side."""
        return cls.WHITE if color == chess.WHITE else cls.BLACK

    @property
    def color(self) -> chess.Color:
        """Return the python-chess color represented by this side."""
        return chess.WHITE if self is ChessSide.WHITE else chess.BLACK


@dataclass(frozen=True)
class SideSubject:
    """One concrete chess side."""

    side: ChessSide


@dataclass(frozen=True)
class PieceSubject:
    """A piece type belonging to one concrete side."""

    piece_type: chess.PieceType
    side: ChessSide

    def __post_init__(self) -> None:
        _validate_piece_type(self.piece_type)


@dataclass(frozen=True)
class SquareSubject:
    """One board square."""

    square: chess.Square

    def __post_init__(self) -> None:
        _validate_square(self.square)


@dataclass(frozen=True)
class MoveSubject:
    """One chess move."""

    move: chess.Move


@dataclass(frozen=True)
class RegionSubject:
    """A named, non-empty collection of board squares."""

    name: str
    squares: tuple[chess.Square, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Region subjects need a non-empty name.")
        if not self.squares:
            raise ValueError("Region subjects need at least one square.")
        for square in self.squares:
            _validate_square(square)
        if len(self.squares) != len(set(self.squares)):
            raise ValueError("Region subject squares must be unique.")


FactSubject: TypeAlias = SideSubject | PieceSubject | SquareSubject | MoveSubject | RegionSubject


@dataclass(frozen=True)
class SupportingPiece:
    """A piece and square that support an evidence value."""

    square: chess.Square
    piece: chess.Piece
    role: str

    def __post_init__(self) -> None:
        _validate_square(self.square)
        if not self.role:
            raise ValueError("Supporting piece roles must not be empty.")


@dataclass(frozen=True)
class AttacksDefendersValue:
    """Counts of opponent attackers and friendly defenders of a square."""

    attackers: int
    defenders: int

    def __post_init__(self) -> None:
        if self.attackers < 0 or self.defenders < 0:
            raise ValueError("Attack and defender counts must be non-negative.")


FactValue: TypeAlias = bool | int | float | str | AttacksDefendersValue


@dataclass(frozen=True)
class AnalyzerProvenance:
    """Stable identity and semantic version of an evidence producer."""

    analyzer: str
    version: str

    def __post_init__(self) -> None:
        if not self.analyzer or not self.version:
            raise ValueError("Analyzer provenance needs a non-empty analyzer and version.")


@dataclass(frozen=True)
class Evidence:
    """One evidence-bearing chess fact.

    ``value`` is absent exactly when ``undefined_reason`` is present.  Supporting
    pieces, squares, and moves are retained even when a downstream consumer only
    needs the scalar value.
    """

    kind: FactKind
    scope: FactScope
    subject: FactSubject
    value: FactValue | None
    perspective: FactPerspective
    guarantee: Guarantee
    provenance: AnalyzerProvenance
    supporting_pieces: tuple[SupportingPiece, ...] = ()
    supporting_squares: tuple[chess.Square, ...] = ()
    supporting_moves: tuple[chess.Move, ...] = ()
    history_requirement: HistoryRequirement = HistoryRequirement.NONE
    history_available: bool = True
    undefined_reason: UndefinedReason | None = None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.undefined_reason is None):
            raise ValueError("Evidence needs exactly one of value or undefined_reason.")
        if self.history_requirement is HistoryRequirement.NONE and not self.history_available:
            raise ValueError("History cannot be unavailable when no history is required.")
        if not self.history_available and self.undefined_reason is not UndefinedReason.HISTORY_UNAVAILABLE:
            raise ValueError("Unavailable required history must make evidence explicitly undefined.")
        if self.history_available and self.undefined_reason is UndefinedReason.HISTORY_UNAVAILABLE:
            raise ValueError("History-unavailable evidence must set history_available to False.")
        for square in self.supporting_squares:
            _validate_square(square)
        if len(self.supporting_squares) != len(set(self.supporting_squares)):
            raise ValueError("Supporting squares must be unique.")
        if len(self.supporting_moves) != len(set(self.supporting_moves)):
            raise ValueError("Supporting moves must be unique.")

    @property
    def is_defined(self) -> bool:
        """Whether the analyzer established a value."""
        return self.undefined_reason is None


class GuaranteeMismatchError(ValueError):
    """Raised when a consumer requests values under a different guarantee."""


@dataclass(frozen=True)
class EvidenceSet:
    """An immutable collection that composes and filters without retagging facts."""

    items: tuple[Evidence, ...] = ()

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def compose(self, *others: Evidence | EvidenceSet) -> EvidenceSet:
        """Return the concatenation of this evidence and other evidence."""
        items = list(self.items)
        for other in others:
            items.extend(other.items if isinstance(other, EvidenceSet) else (other,))
        return EvidenceSet(tuple(items))

    def filter(
        self,
        *,
        kind: FactKind | None = None,
        scope: FactScope | None = None,
        guarantee: Guarantee | None = None,
        predicate: Callable[[Evidence], bool] | None = None,
    ) -> EvidenceSet:
        """Select evidence while preserving each original record unchanged."""
        return EvidenceSet(
            tuple(
                evidence
                for evidence in self.items
                if (kind is None or evidence.kind is kind)
                and (scope is None or evidence.scope is scope)
                and (guarantee is None or evidence.guarantee is guarantee)
                and (predicate is None or predicate(evidence))
            )
        )

    def values(self, *, guarantee: Guarantee) -> tuple[FactValue, ...]:
        """Return defined values only after an explicit guarantee check."""
        mismatches = [item.guarantee for item in self.items if item.guarantee is not guarantee]
        if mismatches:
            raise GuaranteeMismatchError(f"Evidence set contains guarantees other than {guarantee.value}.")
        return tuple(item.value for item in self.items if item.value is not None)


@runtime_checkable
class FactAnalyzer(Protocol):
    """Protocol implemented by chess fact analyzers."""

    kind: FactKind
    scope: FactScope
    guarantee: Guarantee
    provenance: AnalyzerProvenance
    history_requirement: HistoryRequirement
    perspective: FactPerspective

    def analyze(self, board: chess.Board) -> Evidence:
        """Analyze a single position and return one uniform evidence record."""
        ...


def analyze_facts(board: chess.Board, *analyzers: FactAnalyzer) -> EvidenceSet:
    """Run analyzers on one position without introducing batching semantics."""
    _validate_board(board)
    return EvidenceSet(tuple(analyzer.analyze(board) for analyzer in analyzers))


def history_is_available(board: chess.Board, requirement: HistoryRequirement) -> bool:
    """Report whether a board carries the history required by an analyzer.

    A full stack must have the FEN's ply count and be rooted at the standard
    initial position. A position reconstructed from an analysis FEN therefore
    does not silently claim complete game history, even when its counters were
    reset.
    """
    _validate_board(board)
    if requirement is HistoryRequirement.NONE:
        return True
    if requirement is HistoryRequirement.LAST_MOVE:
        return bool(board.move_stack)
    return len(board.move_stack) == board.ply() and board.root().fen() == chess.STARTING_FEN


_REFERENCE_VERSION = "1"


class MaterialAnalyzer:
    """Compute exact material points for one perspective-resolved side."""

    kind = FactKind.MATERIAL
    scope = FactScope.SIDE
    guarantee = Guarantee.EXACT
    provenance = AnalyzerProvenance("lczerolens.reference.material", _REFERENCE_VERSION)
    history_requirement = HistoryRequirement.NONE
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def __init__(self, perspective: FactPerspective = FactPerspective.SIDE_TO_MOVE):
        _validate_side_perspective(perspective)
        self.perspective = perspective

    def analyze(self, board: chess.Board) -> Evidence:
        _validate_board(board)
        side = _resolve_side(board, self.perspective)
        pieces = _pieces_for(board, side.color, "material")
        value = sum(self.piece_values[item.piece.piece_type] for item in pieces)
        return _defined(self, SideSubject(side), value, pieces=pieces)


class PiecePresenceAnalyzer:
    """Determine whether one perspective-resolved side has a piece type."""

    kind = FactKind.PIECE_PRESENCE
    scope = FactScope.PIECE
    guarantee = Guarantee.EXACT
    provenance = AnalyzerProvenance("lczerolens.reference.piece_presence", _REFERENCE_VERSION)
    history_requirement = HistoryRequirement.NONE

    def __init__(
        self,
        piece_type: chess.PieceType,
        perspective: FactPerspective = FactPerspective.SIDE_TO_MOVE,
    ):
        _validate_piece_type(piece_type)
        _validate_side_perspective(perspective)
        self.piece_type = piece_type
        self.perspective = perspective

    def analyze(self, board: chess.Board) -> Evidence:
        _validate_board(board)
        side = _resolve_side(board, self.perspective)
        pieces = tuple(
            SupportingPiece(square, board.piece_at(square), "present")
            for square in sorted(board.pieces(self.piece_type, side.color))
        )
        return _defined(self, PieceSubject(self.piece_type, side), bool(pieces), pieces=pieces)


class AttacksDefendersAnalyzer:
    """Count enemy attackers and friendly defenders of a square."""

    kind = FactKind.ATTACKS_DEFENDERS
    scope = FactScope.SQUARE
    guarantee = Guarantee.EXACT
    provenance = AnalyzerProvenance("lczerolens.reference.attacks_defenders", _REFERENCE_VERSION)
    history_requirement = HistoryRequirement.NONE

    def __init__(
        self,
        square: chess.Square,
        perspective: FactPerspective = FactPerspective.SIDE_TO_MOVE,
    ):
        _validate_square(square)
        _validate_side_perspective(perspective)
        self.square = square
        self.perspective = perspective

    def analyze(self, board: chess.Board) -> Evidence:
        _validate_board(board)
        side = _resolve_side(board, self.perspective)
        attacker_squares = sorted(board.attackers(not side.color, self.square))
        defender_squares = sorted(board.attackers(side.color, self.square))
        pieces = tuple(
            SupportingPiece(square, board.piece_at(square), "attacker") for square in attacker_squares
        ) + tuple(SupportingPiece(square, board.piece_at(square), "defender") for square in defender_squares)
        squares = tuple(dict.fromkeys((self.square, *attacker_squares, *defender_squares)))
        value = AttacksDefendersValue(len(attacker_squares), len(defender_squares))
        return _defined(self, SquareSubject(self.square), value, pieces=pieces, squares=squares)


class CheckStatusAnalyzer:
    """Determine exactly whether a perspective-resolved king is attacked."""

    kind = FactKind.CHECK_STATUS
    scope = FactScope.SIDE
    guarantee = Guarantee.EXACT
    provenance = AnalyzerProvenance("lczerolens.reference.check_status", _REFERENCE_VERSION)
    history_requirement = HistoryRequirement.NONE

    def __init__(self, perspective: FactPerspective = FactPerspective.SIDE_TO_MOVE):
        _validate_side_perspective(perspective)
        self.perspective = perspective

    def analyze(self, board: chess.Board) -> Evidence:
        _validate_board(board)
        side = _resolve_side(board, self.perspective)
        king = board.king(side.color)
        if king is None:
            return _undefined(self, SideSubject(side), UndefinedReason.MISSING_KING)
        attacker_squares = sorted(board.attackers(not side.color, king))
        pieces = tuple(SupportingPiece(square, board.piece_at(square), "checking") for square in attacker_squares)
        squares = tuple((king, *attacker_squares))
        return _defined(self, SideSubject(side), bool(attacker_squares), pieces=pieces, squares=squares)


class LegalMobilityAnalyzer:
    """Count and retain all legal moves for the current side to move."""

    kind = FactKind.LEGAL_MOBILITY
    scope = FactScope.SIDE
    guarantee = Guarantee.EXACT
    provenance = AnalyzerProvenance("lczerolens.reference.legal_mobility", _REFERENCE_VERSION)
    history_requirement = HistoryRequirement.NONE
    perspective = FactPerspective.SIDE_TO_MOVE

    def analyze(self, board: chess.Board) -> Evidence:
        _validate_board(board)
        side = ChessSide.from_color(board.turn)
        if board.status() != chess.STATUS_VALID:
            return _undefined(self, SideSubject(side), UndefinedReason.INVALID_POSITION)
        moves = tuple(board.generate_legal_moves())
        pieces = tuple(SupportingPiece(move.from_square, board.piece_at(move.from_square), "mobile") for move in moves)
        return _defined(self, SideSubject(side), len(moves), pieces=pieces, moves=moves)


def _defined(
    analyzer: FactAnalyzer,
    subject: FactSubject,
    value: FactValue,
    *,
    pieces: tuple[SupportingPiece, ...] = (),
    squares: tuple[chess.Square, ...] = (),
    moves: tuple[chess.Move, ...] = (),
) -> Evidence:
    return Evidence(
        kind=analyzer.kind,
        scope=analyzer.scope,
        subject=subject,
        value=value,
        perspective=analyzer.perspective,
        guarantee=analyzer.guarantee,
        provenance=analyzer.provenance,
        supporting_pieces=pieces,
        supporting_squares=squares,
        supporting_moves=moves,
        history_requirement=analyzer.history_requirement,
    )


def _undefined(analyzer: FactAnalyzer, subject: FactSubject, reason: UndefinedReason) -> Evidence:
    return Evidence(
        kind=analyzer.kind,
        scope=analyzer.scope,
        subject=subject,
        value=None,
        perspective=analyzer.perspective,
        guarantee=analyzer.guarantee,
        provenance=analyzer.provenance,
        history_requirement=analyzer.history_requirement,
        undefined_reason=reason,
    )


def _pieces_for(board: chess.Board, color: chess.Color, role: str) -> tuple[SupportingPiece, ...]:
    return tuple(
        SupportingPiece(square, piece, role)
        for square, piece in sorted(board.piece_map().items())
        if piece.color == color
    )


def _resolve_side(board: chess.Board, perspective: FactPerspective) -> ChessSide:
    if perspective is FactPerspective.SIDE_TO_MOVE:
        return ChessSide.from_color(board.turn)
    if perspective is FactPerspective.WHITE:
        return ChessSide.WHITE
    if perspective is FactPerspective.BLACK:
        return ChessSide.BLACK
    raise ValueError("A side-valued analyzer needs white, black, or side-to-move perspective.")


def _validate_board(board: chess.Board) -> None:
    if not isinstance(board, chess.Board):
        raise TypeError("Fact analyzers require a python-chess Board.")


def _validate_piece_type(piece_type: chess.PieceType) -> None:
    if not isinstance(piece_type, int) or isinstance(piece_type, bool) or piece_type not in chess.PIECE_TYPES:
        raise ValueError("piece_type must be a python-chess piece type.")


def _validate_square(square: chess.Square) -> None:
    if not isinstance(square, int) or isinstance(square, bool) or square not in chess.SQUARES:
        raise ValueError("square must be a python-chess square.")


def _validate_side_perspective(perspective: FactPerspective) -> None:
    if not any(
        perspective is allowed
        for allowed in (FactPerspective.WHITE, FactPerspective.BLACK, FactPerspective.SIDE_TO_MOVE)
    ):
        raise ValueError("A side-valued analyzer needs white, black, or side-to-move perspective.")


__all__ = [
    "AnalyzerProvenance",
    "AttacksDefendersAnalyzer",
    "AttacksDefendersValue",
    "CheckStatusAnalyzer",
    "ChessSide",
    "Evidence",
    "EvidenceSet",
    "FactAnalyzer",
    "FactKind",
    "FactPerspective",
    "FactScope",
    "Guarantee",
    "GuaranteeMismatchError",
    "HistoryRequirement",
    "LegalMobilityAnalyzer",
    "MaterialAnalyzer",
    "MoveSubject",
    "PiecePresenceAnalyzer",
    "PieceSubject",
    "RegionSubject",
    "SideSubject",
    "SquareSubject",
    "SupportingPiece",
    "UndefinedReason",
    "analyze_facts",
    "history_is_available",
]
