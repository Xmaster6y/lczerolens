"""Constraint-aware chess counterfactuals with explicit validity tiers.

Sibling counterfactuals replay two legal moves from one parent. Structural
counterfactuals edit a position directly and can establish rule validity, but
never historical reachability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

import chess

from .facts import Evidence, FactAnalyzer, FactKind, HistoryRequirement, history_is_available
from .move_evidence import MoveDelta, analyze_move_delta


class CounterfactualValidity(str, Enum):
    """Strongest validity claim established for a result."""

    NO_COUNTERFACTUAL = "no_counterfactual"
    RULE_VALID = "rule_valid"
    SIBLING_LEGAL_MOVE = "sibling_legal_move"
    HISTORY_CONSISTENT = "history_consistent"


class PositionAttribute(str, Enum):
    """Rule or metadata attributes that constraints can compare."""

    TURN = "turn"
    KINGS = "kings"
    MATERIAL = "material"
    CASTLING_RIGHTS = "castling_rights"
    EN_PASSANT = "en_passant"
    HALFMOVE_CLOCK = "halfmove_clock"
    HISTORY = "history"


class ConstraintRelation(str, Enum):
    """Requested relation between original and modified positions."""

    CHANGED = "changed"
    PRESERVED = "preserved"


class CounterfactualFailureReason(str, Enum):
    """Stable reasons why no counterfactual was produced."""

    ILLEGAL_FACTUAL_MOVE = "illegal_factual_move"
    ILLEGAL_ALTERNATIVE_MOVE = "illegal_alternative_move"
    NO_ALTERNATIVE = "no_alternative"
    NO_SATISFYING_COUNTERFACTUAL = "no_satisfying_counterfactual"
    EMPTY_SOURCE_SQUARE = "empty_source_square"
    OCCUPIED_TARGET_SQUARE = "occupied_target_square"
    KING_EDIT_FORBIDDEN = "king_edit_forbidden"
    INVALID_POSITION = "invalid_position"
    CONSTRAINT_VIOLATION = "constraint_violation"


@dataclass(frozen=True)
class SiblingMoveOperator:
    """Replace one legal move with another legal move from the same parent."""

    factual_move: chess.Move
    alternative_move: chess.Move | None = None


@dataclass(frozen=True)
class RemovePieceOperator:
    """Remove the non-king piece on ``square``."""

    square: chess.Square

    def __post_init__(self) -> None:
        _validate_square(self.square)


@dataclass(frozen=True)
class RelocatePieceOperator:
    """Relocate a non-king piece to an empty square without making a move."""

    from_square: chess.Square
    to_square: chess.Square

    def __post_init__(self) -> None:
        _validate_square(self.from_square)
        _validate_square(self.to_square)
        if self.from_square == self.to_square:
            raise ValueError("Relocation source and target squares must differ.")


CounterfactualOperator: TypeAlias = SiblingMoveOperator | RemovePieceOperator | RelocatePieceOperator


@dataclass(frozen=True)
class CounterfactualConstraints:
    """Metadata and evidence-bearing fact relations a result must satisfy."""

    changed_attributes: frozenset[PositionAttribute] = frozenset()
    preserved_attributes: frozenset[PositionAttribute] = frozenset()
    changed_facts: tuple[FactAnalyzer, ...] = ()
    preserved_facts: tuple[FactAnalyzer, ...] = ()

    def __post_init__(self) -> None:
        overlap = self.changed_attributes & self.preserved_attributes
        if overlap:
            labels = ", ".join(sorted(item.value for item in overlap))
            raise ValueError(f"Attributes cannot be both changed and preserved: {labels}.")
        for attribute in self.changed_attributes | self.preserved_attributes:
            if not isinstance(attribute, PositionAttribute):
                raise ValueError("Attribute constraints must use PositionAttribute values.")
        for analyzer in (*self.changed_facts, *self.preserved_facts):
            if not isinstance(analyzer, FactAnalyzer):
                raise TypeError("Fact constraints must implement FactAnalyzer.")


@dataclass(frozen=True)
class CounterfactualPosition:
    """Serializable position identity plus rule and history status."""

    fen: str
    status: chess.Status
    rule_valid: bool
    history_complete: bool


@dataclass(frozen=True)
class HistoryGuarantee:
    """What is known about the relationship between the position histories."""

    original_complete: bool
    modified_complete: bool
    shared_parent: bool
    legal_from_shared_parent: bool
    reachability_proven: bool


@dataclass(frozen=True)
class AttributeVerification:
    """Observed values and outcome for one requested metadata constraint."""

    attribute: PositionAttribute
    relation: ConstraintRelation
    original: object
    modified: object
    satisfied: bool


@dataclass(frozen=True)
class AttributeChange:
    """Observed relation for one metadata attribute, whether constrained or not."""

    attribute: PositionAttribute
    original: object
    modified: object
    changed: bool


@dataclass(frozen=True)
class FactVerification:
    """Original evidence and modified evidence for one requested fact relation."""

    relation: ConstraintRelation
    original: Evidence
    modified: Evidence
    satisfied: bool


@dataclass(frozen=True)
class CounterfactualFailure:
    """One machine-readable failure with optional candidate context."""

    reason: CounterfactualFailureReason
    message: str
    candidate_move: chess.Move | None = None
    attribute: PositionAttribute | None = None
    analyzer: str | None = None


@dataclass(frozen=True)
class CounterfactualResult:
    """A successful intervention or structured explanation of its failure."""

    original: CounterfactualPosition
    modified: CounterfactualPosition | None
    operator: CounterfactualOperator
    validity: CounterfactualValidity
    history: HistoryGuarantee
    attributes: tuple[AttributeChange, ...] = ()
    changed_attributes: tuple[AttributeVerification, ...] = ()
    preserved_attributes: tuple[AttributeVerification, ...] = ()
    changed_facts: tuple[FactVerification, ...] = ()
    preserved_facts: tuple[FactVerification, ...] = ()
    original_move_delta: MoveDelta | None = None
    modified_move_delta: MoveDelta | None = None
    failures: tuple[CounterfactualFailure, ...] = ()

    @property
    def succeeded(self) -> bool:
        """Whether a satisfying, valid modified position was produced."""
        return self.modified is not None and not self.failures


def sibling_counterfactual(
    parent: chess.Board,
    factual_move: chess.Move,
    alternative_move: chess.Move | None = None,
    *,
    constraints: CounterfactualConstraints | None = None,
) -> CounterfactualResult:
    """Compare two legal children of one parent, choosing deterministically.

    When ``alternative_move`` is omitted, legal alternatives are tried in UCI
    order and the first one satisfying all constraints is returned.
    """
    _validate_board(parent)
    resolved = constraints or CounterfactualConstraints()
    operator = SiblingMoveOperator(factual_move, alternative_move)
    parent_snapshot = _snapshot(parent)
    failure_history = _history_guarantee(parent, None, shared_parent=False, legal=False)
    if not parent.is_valid():
        return _failure_result(
            parent_snapshot,
            operator,
            failure_history,
            CounterfactualFailureReason.INVALID_POSITION,
            f"The supplied parent position has invalid python-chess status {int(parent.status())}.",
        )
    if not _is_legal_move(parent, factual_move):
        return _failure_result(
            parent_snapshot,
            operator,
            failure_history,
            CounterfactualFailureReason.ILLEGAL_FACTUAL_MOVE,
            "The factual move is not legal in the supplied parent position.",
            candidate_move=factual_move if isinstance(factual_move, chess.Move) else None,
        )

    factual_board = parent.copy(stack=True)
    factual_board.push(factual_move)
    original = _snapshot(factual_board)
    candidates = (
        (alternative_move,)
        if alternative_move is not None
        else tuple(sorted((move for move in parent.legal_moves if move != factual_move), key=lambda move: move.uci()))
    )
    if not candidates:
        return _failure_result(
            original,
            operator,
            _history_guarantee(factual_board, None, shared_parent=True, legal=True),
            CounterfactualFailureReason.NO_ALTERNATIVE,
            "The parent position has no legal move other than the factual move.",
        )

    candidate_failures: list[CounterfactualFailure] = []
    for candidate in candidates:
        if not _is_legal_move(parent, candidate) or candidate == factual_move:
            candidate_failures.append(
                CounterfactualFailure(
                    CounterfactualFailureReason.ILLEGAL_ALTERNATIVE_MOVE,
                    "The alternative must be a different legal move from the same parent.",
                    candidate_move=candidate if isinstance(candidate, chess.Move) else None,
                )
            )
            continue
        modified_board = parent.copy(stack=True)
        modified_board.push(candidate)
        verifications, failures = _verify_constraints(factual_board, modified_board, resolved, candidate)
        if failures:
            candidate_failures.extend(failures)
            continue
        complete = history_is_available(modified_board, HistoryRequirement.FULL_MOVE_STACK)
        return _success_result(
            factual_board,
            modified_board,
            SiblingMoveOperator(factual_move, candidate),
            CounterfactualValidity.HISTORY_CONSISTENT if complete else CounterfactualValidity.SIBLING_LEGAL_MOVE,
            _history_guarantee(factual_board, modified_board, shared_parent=True, legal=True),
            verifications,
            original_move_delta=analyze_move_delta(parent, factual_move),
            modified_move_delta=analyze_move_delta(parent, candidate),
        )

    if alternative_move is not None:
        failures = tuple(candidate_failures)
    else:
        reason = CounterfactualFailureReason.NO_SATISFYING_COUNTERFACTUAL
        message = f"None of {len(candidates)} deterministic legal alternatives satisfied the constraints."
        failures = (CounterfactualFailure(reason, message), *candidate_failures)
    return CounterfactualResult(
        original=original,
        modified=None,
        operator=operator,
        validity=CounterfactualValidity.NO_COUNTERFACTUAL,
        history=_history_guarantee(factual_board, None, shared_parent=True, legal=True),
        failures=failures,
    )


def remove_piece_counterfactual(
    board: chess.Board,
    square: chess.Square,
    *,
    constraints: CounterfactualConstraints | None = None,
) -> CounterfactualResult:
    """Remove a non-king piece and return a rule-valid structural state."""
    _validate_board(board)
    operator = RemovePieceOperator(square)
    if not board.is_valid():
        return _structural_failure(
            board,
            operator,
            CounterfactualFailureReason.INVALID_POSITION,
            f"The supplied position has invalid python-chess status {int(board.status())}.",
        )
    piece = board.piece_at(square)
    if piece is None:
        return _structural_failure(
            board, operator, CounterfactualFailureReason.EMPTY_SOURCE_SQUARE, "No piece exists on the source square."
        )
    if piece.piece_type == chess.KING:
        return _structural_failure(
            board,
            operator,
            CounterfactualFailureReason.KING_EDIT_FORBIDDEN,
            "Structural operators cannot remove a king.",
        )
    modified = board.copy(stack=False)
    modified.remove_piece_at(square)
    _normalize_structural_metadata(modified)
    return _finish_structural(board, modified, operator, constraints or CounterfactualConstraints())


def relocate_piece_counterfactual(
    board: chess.Board,
    from_square: chess.Square,
    to_square: chess.Square,
    *,
    constraints: CounterfactualConstraints | None = None,
) -> CounterfactualResult:
    """Relocate a non-king piece and return a rule-valid structural state."""
    _validate_board(board)
    operator = RelocatePieceOperator(from_square, to_square)
    if not board.is_valid():
        return _structural_failure(
            board,
            operator,
            CounterfactualFailureReason.INVALID_POSITION,
            f"The supplied position has invalid python-chess status {int(board.status())}.",
        )
    piece = board.piece_at(from_square)
    if piece is None:
        return _structural_failure(
            board, operator, CounterfactualFailureReason.EMPTY_SOURCE_SQUARE, "No piece exists on the source square."
        )
    if board.piece_at(to_square) is not None:
        return _structural_failure(
            board, operator, CounterfactualFailureReason.OCCUPIED_TARGET_SQUARE, "The relocation target must be empty."
        )
    if piece.piece_type == chess.KING:
        return _structural_failure(
            board,
            operator,
            CounterfactualFailureReason.KING_EDIT_FORBIDDEN,
            "Structural operators cannot relocate a king.",
        )
    modified = board.copy(stack=False)
    modified.remove_piece_at(from_square)
    modified.set_piece_at(to_square, piece, promoted=bool(board.promoted & chess.BB_SQUARES[from_square]))
    _normalize_structural_metadata(modified)
    return _finish_structural(board, modified, operator, constraints or CounterfactualConstraints())


def _finish_structural(
    original: chess.Board,
    modified: chess.Board,
    operator: CounterfactualOperator,
    constraints: CounterfactualConstraints,
) -> CounterfactualResult:
    history = _history_guarantee(original, modified, shared_parent=False, legal=False)
    if not modified.is_valid():
        return _failure_result(
            _snapshot(original),
            operator,
            history,
            CounterfactualFailureReason.INVALID_POSITION,
            f"The structural edit produced invalid python-chess status {int(modified.status())}.",
        )
    verifications, failures = _verify_constraints(original, modified, constraints)
    if failures:
        return CounterfactualResult(
            original=_snapshot(original),
            modified=None,
            operator=operator,
            validity=CounterfactualValidity.NO_COUNTERFACTUAL,
            history=history,
            attributes=_compare_attributes(original, modified),
            changed_attributes=verifications[0],
            preserved_attributes=verifications[1],
            changed_facts=verifications[2],
            preserved_facts=verifications[3],
            failures=failures,
        )
    return _success_result(
        original,
        modified,
        operator,
        CounterfactualValidity.RULE_VALID,
        history,
        verifications,
    )


VerificationGroups: TypeAlias = tuple[
    tuple[AttributeVerification, ...],
    tuple[AttributeVerification, ...],
    tuple[FactVerification, ...],
    tuple[FactVerification, ...],
]


def _verify_constraints(
    original: chess.Board,
    modified: chess.Board,
    constraints: CounterfactualConstraints,
    candidate_move: chess.Move | None = None,
) -> tuple[VerificationGroups, tuple[CounterfactualFailure, ...]]:
    changed_attributes = tuple(
        _verify_attribute(original, modified, attribute, ConstraintRelation.CHANGED)
        for attribute in sorted(constraints.changed_attributes, key=lambda item: item.value)
    )
    preserved_attributes = tuple(
        _verify_attribute(original, modified, attribute, ConstraintRelation.PRESERVED)
        for attribute in sorted(constraints.preserved_attributes, key=lambda item: item.value)
    )
    changed_facts = tuple(
        _verify_fact(original, modified, analyzer, ConstraintRelation.CHANGED)
        for analyzer in constraints.changed_facts
    )
    preserved_facts = tuple(
        _verify_fact(original, modified, analyzer, ConstraintRelation.PRESERVED)
        for analyzer in constraints.preserved_facts
    )
    failures: list[CounterfactualFailure] = []
    for verification in (*changed_attributes, *preserved_attributes):
        if not verification.satisfied:
            failures.append(
                CounterfactualFailure(
                    CounterfactualFailureReason.CONSTRAINT_VIOLATION,
                    f"Attribute {verification.attribute.value} was not {verification.relation.value}.",
                    candidate_move=candidate_move,
                    attribute=verification.attribute,
                )
            )
    for verification in (*changed_facts, *preserved_facts):
        if not verification.satisfied:
            failures.append(
                CounterfactualFailure(
                    CounterfactualFailureReason.CONSTRAINT_VIOLATION,
                    f"Fact {verification.original.provenance.analyzer} was not {verification.relation.value}.",
                    candidate_move=candidate_move,
                    analyzer=verification.original.provenance.analyzer,
                )
            )
    return (changed_attributes, preserved_attributes, changed_facts, preserved_facts), tuple(failures)


def _verify_attribute(
    original: chess.Board,
    modified: chess.Board,
    attribute: PositionAttribute,
    relation: ConstraintRelation,
) -> AttributeVerification:
    before = _attribute_value(original, attribute)
    after = _attribute_value(modified, attribute)
    equal = before == after
    return AttributeVerification(
        attribute, relation, before, after, equal is (relation is ConstraintRelation.PRESERVED)
    )


def _verify_fact(
    original: chess.Board,
    modified: chess.Board,
    analyzer: FactAnalyzer,
    relation: ConstraintRelation,
) -> FactVerification:
    before = analyzer.analyze(original)
    after = analyzer.analyze(modified)
    equal = _same_evidence(before, after)
    return FactVerification(relation, before, after, equal is (relation is ConstraintRelation.PRESERVED))


def _compare_attributes(original: chess.Board, modified: chess.Board) -> tuple[AttributeChange, ...]:
    comparisons: list[AttributeChange] = []
    for attribute in PositionAttribute:
        before = _attribute_value(original, attribute)
        after = _attribute_value(modified, attribute)
        comparisons.append(AttributeChange(attribute, before, after, before != after))
    return tuple(comparisons)


def _same_evidence(original: Evidence, modified: Evidence) -> bool:
    same_value = (
        original.kind == modified.kind
        and original.scope == modified.scope
        and original.subject == modified.subject
        and original.value == modified.value
        and original.perspective == modified.perspective
        and original.guarantee == modified.guarantee
        and original.provenance == modified.provenance
        and original.history_requirement == modified.history_requirement
        and original.history_available == modified.history_available
        and original.undefined_reason == modified.undefined_reason
    )
    if not same_value:
        return False
    if original.kind in (FactKind.ATTACKS_DEFENDERS, FactKind.CHECK_STATUS):
        return (
            original.supporting_pieces == modified.supporting_pieces
            and original.supporting_squares == modified.supporting_squares
        )
    return True


def _attribute_value(board: chess.Board, attribute: PositionAttribute) -> object:
    if attribute is PositionAttribute.TURN:
        return board.turn
    if attribute is PositionAttribute.KINGS:
        return (board.king(chess.WHITE), board.king(chess.BLACK))
    if attribute is PositionAttribute.MATERIAL:
        return tuple(
            len(board.pieces(piece_type, color)) for color in chess.COLORS for piece_type in chess.PIECE_TYPES
        )
    if attribute is PositionAttribute.CASTLING_RIGHTS:
        return board.castling_rights
    if attribute is PositionAttribute.EN_PASSANT:
        return board.ep_square
    if attribute is PositionAttribute.HALFMOVE_CLOCK:
        return board.halfmove_clock
    if attribute is PositionAttribute.HISTORY:
        return history_is_available(board, HistoryRequirement.FULL_MOVE_STACK)
    raise AssertionError(f"Unhandled position attribute: {attribute!r}.")


def _success_result(
    original: chess.Board,
    modified: chess.Board,
    operator: CounterfactualOperator,
    validity: CounterfactualValidity,
    history: HistoryGuarantee,
    verifications: VerificationGroups,
    *,
    original_move_delta: MoveDelta | None = None,
    modified_move_delta: MoveDelta | None = None,
) -> CounterfactualResult:
    return CounterfactualResult(
        original=_snapshot(original),
        modified=_snapshot(modified),
        operator=operator,
        validity=validity,
        history=history,
        attributes=_compare_attributes(original, modified),
        changed_attributes=verifications[0],
        preserved_attributes=verifications[1],
        changed_facts=verifications[2],
        preserved_facts=verifications[3],
        original_move_delta=original_move_delta,
        modified_move_delta=modified_move_delta,
    )


def _failure_result(
    original: CounterfactualPosition,
    operator: CounterfactualOperator,
    history: HistoryGuarantee,
    reason: CounterfactualFailureReason,
    message: str,
    *,
    candidate_move: chess.Move | None = None,
) -> CounterfactualResult:
    return CounterfactualResult(
        original=original,
        modified=None,
        operator=operator,
        validity=CounterfactualValidity.NO_COUNTERFACTUAL,
        history=history,
        failures=(CounterfactualFailure(reason, message, candidate_move=candidate_move),),
    )


def _structural_failure(
    board: chess.Board,
    operator: CounterfactualOperator,
    reason: CounterfactualFailureReason,
    message: str,
) -> CounterfactualResult:
    return _failure_result(
        _snapshot(board),
        operator,
        _history_guarantee(board, None, shared_parent=False, legal=False),
        reason,
        message,
    )


def _snapshot(board: chess.Board) -> CounterfactualPosition:
    return CounterfactualPosition(
        fen=board.fen(en_passant="fen"),
        status=board.status(),
        rule_valid=board.is_valid(),
        history_complete=history_is_available(board, HistoryRequirement.FULL_MOVE_STACK),
    )


def _history_guarantee(
    original: chess.Board,
    modified: chess.Board | None,
    *,
    shared_parent: bool,
    legal: bool,
) -> HistoryGuarantee:
    original_complete = history_is_available(original, HistoryRequirement.FULL_MOVE_STACK)
    modified_complete = modified is not None and history_is_available(modified, HistoryRequirement.FULL_MOVE_STACK)
    return HistoryGuarantee(
        original_complete=original_complete,
        modified_complete=modified_complete,
        shared_parent=shared_parent,
        legal_from_shared_parent=legal,
        reachability_proven=original_complete and modified_complete and shared_parent and legal,
    )


def _normalize_structural_metadata(board: chess.Board) -> None:
    board.castling_rights = board.clean_castling_rights()
    if board.status() & chess.STATUS_INVALID_EP_SQUARE:
        board.ep_square = None


def _validate_board(board: chess.Board) -> None:
    if not isinstance(board, chess.Board):
        raise TypeError("Counterfactual generation requires a python-chess Board.")


def _validate_square(square: chess.Square) -> None:
    if not isinstance(square, int) or isinstance(square, bool) or square not in chess.SQUARES:
        raise ValueError(f"Invalid chess square: {square!r}.")


def _is_legal_move(board: chess.Board, move: object) -> bool:
    return (
        isinstance(move, chess.Move)
        and move.from_square in chess.SQUARES
        and move.to_square in chess.SQUARES
        and move in board.legal_moves
    )
