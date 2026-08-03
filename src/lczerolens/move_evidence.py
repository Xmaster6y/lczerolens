"""Exact move deltas and variation evidence built from chess facts.

This module compares evidence records across legal moves.  It deliberately
describes only rule-exact changes and retains the original evidence objects;
interpretations such as initiative or compensation belong to downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import chess

from .facts import (
    AttacksDefendersAnalyzer,
    CheckStatusAnalyzer,
    ChessSide,
    Evidence,
    EvidenceSet,
    FactAnalyzer,
    FactKind,
    FactPerspective,
    HistoryRequirement,
    LegalMobilityAnalyzer,
    MaterialAnalyzer,
    PiecePresenceAnalyzer,
    analyze_facts,
    history_is_available,
)


class ExactMoveEffect(str, Enum):
    """Rule-exact effects established directly from a legal move."""

    CAPTURE = "capture"
    CHECK = "check"
    EVASION = "evasion"
    PROMOTION = "promotion"
    CASTLING = "castling"
    EN_PASSANT = "en_passant"


class EvidenceTransitionKind(str, Enum):
    """Whether one semantic fact was retained or changed."""

    PRESERVED = "preserved"
    CHANGED = "changed"


class HistoryPolicy(str, Enum):
    """How variation analysis treats a truncated input move stack."""

    ALLOW_TRUNCATED = "allow_truncated"
    REQUIRE_COMPLETE = "require_complete"


class VariationRole(str, Enum):
    """Machine-readable purpose of a line relative to a claim."""

    NEUTRAL = "neutral"
    CANDIDATE_SUPPORT = "candidate_support"
    OPPONENT_REFUTATION = "opponent_refutation"


class VariationFailureReason(str, Enum):
    """Structured reasons why a variation could not be analyzed."""

    EMPTY_LINE = "empty_line"
    ILLEGAL_MOVE = "illegal_move"
    HISTORY_INCOMPATIBLE = "history_incompatible"
    INTENT_MISMATCH = "intent_mismatch"


@dataclass(frozen=True)
class PositionEvidence:
    """Stable position identity and the facts observed there."""

    fen: str
    turn: ChessSide
    ply: int
    history_complete: bool
    evidence: EvidenceSet

    @property
    def history_truncated(self) -> bool:
        """Whether the position lacks a complete standard-game move stack."""
        return not self.history_complete


@dataclass(frozen=True)
class MoveEvidence:
    """Exact position and chess-object provenance for one legal move."""

    move: chess.Move
    mover: ChessSide
    moving_piece: chess.Piece
    captured_piece: chess.Piece | None
    capture_square: chess.Square | None
    rook_move: chess.Move | None
    effects: tuple[ExactMoveEffect, ...]


@dataclass(frozen=True)
class EvidenceTransition:
    """The before/after evidence records for one semantic fact."""

    before: Evidence
    after: Evidence
    kind: EvidenceTransitionKind

    def __post_init__(self) -> None:
        same_value = _same_fact_value(self.before, self.after)
        if self.kind is EvidenceTransitionKind.PRESERVED and not same_value:
            raise ValueError("Preserved transitions require the same fact identity and value.")
        if self.kind is EvidenceTransitionKind.CHANGED and same_value:
            raise ValueError("Changed transitions require a different fact identity or value.")


@dataclass(frozen=True)
class MoveDelta:
    """Deterministic fact-set changes caused by one legal move.

    ``created`` and ``removed`` are the mathematical set-difference views.  A
    changed semantic fact contributes its after record to ``created`` and its
    before record to ``removed``.  ``transitions`` links both original records.
    """

    before: PositionEvidence
    after: PositionEvidence
    move: MoveEvidence
    created: EvidenceSet
    removed: EvidenceSet
    preserved: EvidenceSet
    transitions: tuple[EvidenceTransition, ...]

    @property
    def changed(self) -> tuple[EvidenceTransition, ...]:
        """Return transitions whose fact value or defined state changed."""
        return tuple(item for item in self.transitions if item.kind is EvidenceTransitionKind.CHANGED)


@dataclass(frozen=True)
class VariationIntent:
    """A structured claim relation for a candidate line."""

    role: VariationRole = VariationRole.NEUTRAL
    claim_id: str | None = None
    candidate: chess.Move | None = None
    response: chess.Move | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, VariationRole):
            raise ValueError("Variation intent role must be a VariationRole.")
        if self.role is VariationRole.NEUTRAL:
            if any(item is not None for item in (self.claim_id, self.candidate, self.response)):
                raise ValueError("Neutral variation intent cannot name a claim, candidate, or response.")
            return
        if not self.claim_id or self.candidate is None:
            raise ValueError("Claim-related variation intent needs a claim_id and candidate move.")
        if self.role is VariationRole.CANDIDATE_SUPPORT and self.response is not None:
            raise ValueError("Candidate-support intent does not take a refuting response.")
        if self.role is VariationRole.OPPONENT_REFUTATION and self.response is None:
            raise ValueError("Opponent-refutation intent needs the opponent response.")


@dataclass(frozen=True)
class VariationTerminal:
    """Rule result at the end of an analyzed line."""

    is_terminal: bool
    result: str
    winner: ChessSide | None
    termination: chess.Termination | None
    claimable_draw: bool


@dataclass(frozen=True)
class VariationEvidence:
    """Evidence for an ordered legal line, one exact delta per ply."""

    initial: PositionEvidence
    final: PositionEvidence
    deltas: tuple[MoveDelta, ...]
    intent: VariationIntent
    history_policy: HistoryPolicy
    terminal: VariationTerminal

    @property
    def moves(self) -> tuple[chess.Move, ...]:
        """Return the analyzed move sequence."""
        return tuple(delta.move.move for delta in self.deltas)


class VariationAnalysisError(ValueError):
    """A variation failure with stable machine-readable context."""

    def __init__(
        self,
        reason: VariationFailureReason,
        message: str,
        *,
        ply_index: int | None = None,
        move: chess.Move | None = None,
        fen: str | None = None,
    ):
        super().__init__(message)
        self.reason = reason
        self.ply_index = ply_index
        self.move = move
        self.fen = fen


def exact_move_analyzers() -> tuple[FactAnalyzer, ...]:
    """Return the complete bundled exact analyzer suite for move comparison.

    Both concrete sides are analyzed where meaningful.  Attack/defender facts
    cover every square from both sides' perspectives; legal mobility remains a
    side-to-move fact by definition.
    """
    analyzers: list[FactAnalyzer] = []
    for perspective in (FactPerspective.WHITE, FactPerspective.BLACK):
        analyzers.extend(
            (
                MaterialAnalyzer(perspective),
                CheckStatusAnalyzer(perspective),
                *(PiecePresenceAnalyzer(piece_type, perspective) for piece_type in chess.PIECE_TYPES),
                *(AttacksDefendersAnalyzer(square, perspective) for square in chess.SQUARES),
            )
        )
    analyzers.append(LegalMobilityAnalyzer())
    return tuple(analyzers)


def analyze_move_delta(
    board: chess.Board,
    move: chess.Move,
    *analyzers: FactAnalyzer,
) -> MoveDelta:
    """Apply one legal move and compare exact evidence before and after it."""
    _validate_board_and_move(board, move)
    selected = tuple(analyzers) or exact_move_analyzers()
    before_set = analyze_facts(board, *selected)
    move_evidence = _move_evidence(board, move)
    after_board = board.copy(stack=True)
    after_board.push(move)
    after_set = analyze_facts(after_board, *selected)
    created, removed, preserved, transitions = _compare_evidence(before_set, after_set)
    return MoveDelta(
        before=_position_evidence(board, before_set),
        after=_position_evidence(after_board, after_set),
        move=move_evidence,
        created=created,
        removed=removed,
        preserved=preserved,
        transitions=transitions,
    )


def analyze_variation(
    board: chess.Board,
    moves: Iterable[chess.Move],
    *analyzers: FactAnalyzer,
    intent: VariationIntent | None = None,
    history_policy: HistoryPolicy = HistoryPolicy.ALLOW_TRUNCATED,
) -> VariationEvidence:
    """Analyze a non-empty ordered legal line with explicit history policy."""
    if not isinstance(board, chess.Board):
        raise TypeError("Variation analysis requires a python-chess Board.")
    if not isinstance(history_policy, HistoryPolicy):
        raise ValueError("history_policy must be a HistoryPolicy.")
    line = tuple(moves)
    if not line:
        raise VariationAnalysisError(VariationFailureReason.EMPTY_LINE, "A variation needs at least one move.")
    complete = history_is_available(board, HistoryRequirement.FULL_MOVE_STACK)
    if history_policy is HistoryPolicy.REQUIRE_COMPLETE and not complete:
        raise VariationAnalysisError(
            VariationFailureReason.HISTORY_INCOMPATIBLE,
            "The variation requires a complete move stack rooted at the standard initial position.",
            fen=board.fen(),
        )
    resolved_intent = intent or VariationIntent()
    _validate_intent_line(resolved_intent, line, board.fen())
    selected = tuple(analyzers) or exact_move_analyzers()
    current = board.copy(stack=True)
    initial_set = analyze_facts(current, *selected)
    initial = _position_evidence(current, initial_set)
    deltas: list[MoveDelta] = []
    for ply_index, move in enumerate(line):
        if not _is_legal_move(current, move):
            raise VariationAnalysisError(
                VariationFailureReason.ILLEGAL_MOVE,
                f"Illegal variation move at ply {ply_index}: {_move_label(move)}.",
                ply_index=ply_index,
                move=move if isinstance(move, chess.Move) else None,
                fen=current.fen(),
            )
        delta = analyze_move_delta(current, move, *selected)
        deltas.append(delta)
        current.push(move)
    final = deltas[-1].after
    outcome = current.outcome()
    terminal = VariationTerminal(
        is_terminal=outcome is not None,
        result=outcome.result() if outcome is not None else "*",
        winner=ChessSide.from_color(outcome.winner) if outcome is not None and outcome.winner is not None else None,
        termination=outcome.termination if outcome is not None else None,
        claimable_draw=(
            complete
            and outcome is None
            and (current.can_claim_fifty_moves() or current.can_claim_threefold_repetition())
        ),
    )
    return VariationEvidence(
        initial=initial,
        final=final,
        deltas=tuple(deltas),
        intent=resolved_intent,
        history_policy=history_policy,
        terminal=terminal,
    )


def _compare_evidence(
    before: EvidenceSet,
    after: EvidenceSet,
) -> tuple[EvidenceSet, EvidenceSet, EvidenceSet, tuple[EvidenceTransition, ...]]:
    unmatched_after = list(after.items)
    preserved: list[Evidence] = []
    changed: list[EvidenceTransition] = []
    removed_only: list[Evidence] = []
    created_only: list[Evidence] = []
    for before_item in before:
        key_index = next(
            (index for index, item in enumerate(unmatched_after) if _evidence_key(item) == _evidence_key(before_item)),
            None,
        )
        if key_index is None:
            removed_only.append(before_item)
            continue
        after_item = unmatched_after.pop(key_index)
        if _same_fact_value(before_item, after_item):
            preserved.append(before_item)
            changed.append(EvidenceTransition(before_item, after_item, EvidenceTransitionKind.PRESERVED))
        else:
            changed.append(EvidenceTransition(before_item, after_item, EvidenceTransitionKind.CHANGED))
            removed_only.append(before_item)
            created_only.append(after_item)
    created_only.extend(unmatched_after)
    return (
        EvidenceSet(tuple(created_only)),
        EvidenceSet(tuple(removed_only)),
        EvidenceSet(tuple(preserved)),
        tuple(changed),
    )


def _evidence_key(evidence: Evidence) -> tuple[object, ...]:
    return (
        evidence.kind,
        evidence.scope,
        evidence.subject,
        evidence.perspective,
        evidence.guarantee,
        evidence.provenance,
        evidence.history_requirement,
    )


def _same_fact_value(before: Evidence, after: Evidence) -> bool:
    same_value = (
        _evidence_key(before) == _evidence_key(after)
        and before.value == after.value
        and before.undefined_reason is after.undefined_reason
        and before.history_available is after.history_available
    )
    if not same_value:
        return False
    if before.kind in (FactKind.ATTACKS_DEFENDERS, FactKind.CHECK_STATUS):
        return (
            before.supporting_pieces == after.supporting_pieces
            and before.supporting_squares == after.supporting_squares
        )
    return True


def _position_evidence(board: chess.Board, evidence: EvidenceSet) -> PositionEvidence:
    return PositionEvidence(
        fen=board.fen(),
        turn=ChessSide.from_color(board.turn),
        ply=board.ply(),
        history_complete=history_is_available(board, HistoryRequirement.FULL_MOVE_STACK),
        evidence=evidence,
    )


def _move_evidence(board: chess.Board, move: chess.Move) -> MoveEvidence:
    moving_piece = board.piece_at(move.from_square)
    assert moving_piece is not None
    is_castling = board.is_castling(move)
    en_passant = board.is_en_passant(move)
    capture_square: chess.Square | None = None
    captured_piece: chess.Piece | None = None
    if not is_castling:
        capture_square = move.to_square
        if en_passant:
            capture_square += -8 if board.turn == chess.WHITE else 8
        captured_piece = board.piece_at(capture_square)
    rook_move = _castling_rook_move(board, move) if is_castling else None
    was_check = board.is_check()
    after = board.copy(stack=False)
    after.push(move)
    effects: list[ExactMoveEffect] = []
    if captured_piece is not None:
        effects.append(ExactMoveEffect.CAPTURE)
    if after.is_check():
        effects.append(ExactMoveEffect.CHECK)
    if was_check:
        effects.append(ExactMoveEffect.EVASION)
    if move.promotion is not None:
        effects.append(ExactMoveEffect.PROMOTION)
    if is_castling:
        effects.append(ExactMoveEffect.CASTLING)
    if en_passant:
        effects.append(ExactMoveEffect.EN_PASSANT)
    return MoveEvidence(
        move=move,
        mover=ChessSide.from_color(board.turn),
        moving_piece=moving_piece,
        captured_piece=captured_piece,
        capture_square=capture_square if captured_piece is not None else None,
        rook_move=rook_move,
        effects=tuple(effects),
    )


def _castling_rook_move(board: chess.Board, move: chess.Move) -> chess.Move:
    rank = chess.square_rank(move.from_square)
    kingside = board.is_kingside_castling(move)
    if board.chess960:
        rook_from = move.to_square
    elif kingside:
        rook_from = chess.square(7, rank)
    else:
        rook_from = chess.square(0, rank)
    rook_to = chess.square(5 if kingside else 3, rank)
    return chess.Move(rook_from, rook_to)


def _validate_board_and_move(board: chess.Board, move: chess.Move) -> None:
    if not isinstance(board, chess.Board):
        raise TypeError("Move-delta analysis requires a python-chess Board.")
    if not _is_legal_move(board, move):
        raise VariationAnalysisError(
            VariationFailureReason.ILLEGAL_MOVE,
            f"Move is not legal in the supplied position: {_move_label(move)}.",
            move=move if isinstance(move, chess.Move) else None,
            fen=board.fen(),
        )


def _is_well_formed_move(move: object) -> bool:
    return isinstance(move, chess.Move) and move.from_square in chess.SQUARES and move.to_square in chess.SQUARES


def _is_legal_move(board: chess.Board, move: object) -> bool:
    return _is_well_formed_move(move) and move in board.legal_moves


def _move_label(move: object) -> str:
    if not _is_well_formed_move(move):
        return "<malformed move>"
    return chess.square_name(move.from_square) + chess.square_name(move.to_square)


def _validate_intent_line(intent: VariationIntent, line: tuple[chess.Move, ...], fen: str) -> None:
    mismatch = False
    if intent.role is VariationRole.CANDIDATE_SUPPORT:
        mismatch = line[0] != intent.candidate
    elif intent.role is VariationRole.OPPONENT_REFUTATION:
        mismatch = len(line) < 2 or line[0] != intent.candidate or line[1] != intent.response
    if mismatch:
        raise VariationAnalysisError(
            VariationFailureReason.INTENT_MISMATCH,
            "The analyzed line does not match its declared candidate/response intent.",
            fen=fen,
        )


__all__ = [
    "EvidenceTransition",
    "EvidenceTransitionKind",
    "ExactMoveEffect",
    "HistoryPolicy",
    "MoveDelta",
    "MoveEvidence",
    "PositionEvidence",
    "VariationAnalysisError",
    "VariationEvidence",
    "VariationFailureReason",
    "VariationIntent",
    "VariationRole",
    "VariationTerminal",
    "analyze_move_delta",
    "analyze_variation",
    "exact_move_analyzers",
]
