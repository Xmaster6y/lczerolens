"""A small deterministic neural-MCTS implementation for auditable experiments.

This module deliberately does not model lc0's batching, virtual visits, tree
reuse, transpositions, pruning, or FPU behaviour.  It emits the public
``SearchTrace`` schema after every sequential simulation instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Protocol

import chess
import torch
from tensordict import TensorDict

from lczerolens.board import LczeroBoard
from lczerolens.search_trace import (
    BackupUpdate,
    ChessPlayer,
    EdgeStatistics,
    EvaluatorCall,
    LeafRecord,
    NodeExpansion,
    PathStep,
    PositionEvaluation,
    RootAction,
    RootSelection,
    RootSnapshot,
    SearchBudget,
    SearchBudgetUnit,
    SearchCapability,
    SearchParameter,
    SearchProvenance,
    SearchTrace,
    SimulationEvent,
    ValuePerspective,
)


class Evaluator(Protocol):
    """The stable #130 evaluator shape consumed by reference search."""

    def __call__(self, board: LczeroBoard) -> TensorDict: ...


@dataclass
class _Edge:
    move: chess.Move
    prior: float
    visits: int = 0
    total_value: float = 0.0
    child: _Node | None = None

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


@dataclass
class _Node:
    board: LczeroBoard
    node_id: str
    edges: dict[str, _Edge] = field(default_factory=dict)
    expanded: bool = False


class ReferenceMCTS:
    """Sequential PUCT search whose output is a replayable :class:`SearchTrace`.

    The evaluator returns a single-board TensorDict with a raw 1858-logit
    ``policy`` and scalar ``value`` for the side to move.  Legal logits are
    masked and softmaxed using :meth:`LczeroBoard.get_legal_policy`.
    """

    def __init__(self, c_puct: float = 1.0):
        if not math.isfinite(c_puct) or c_puct < 0:
            raise ValueError("c_puct must be finite and non-negative.")
        self.c_puct = c_puct

    def search(
        self,
        board: LczeroBoard,
        evaluator: Evaluator | Callable[[LczeroBoard], TensorDict],
        simulations: int,
    ) -> SearchTrace:
        """Run a fixed number of simulations and return their full trace."""
        if not isinstance(simulations, int) or isinstance(simulations, bool) or simulations < 1:
            raise ValueError("simulations must be a positive integer.")
        if board.is_game_over():
            raise ValueError("Reference search requires a non-terminal root position.")

        root = _Node(board.copy(stack=True), "node-0")
        node_count = 1
        root_evaluation, _, _ = self._expand(root, evaluator, ValuePerspective.ROOT_PLAYER)
        events: list[SimulationEvent] = []

        for simulation in range(simulations):
            root_before = self._edge_stats(root, ValuePerspective.ROOT_PLAYER)
            node = root
            path: list[tuple[_Node, _Edge, _Node]] = []
            expansion: NodeExpansion | None = None

            while node.expanded and node.edges:
                edge = self._select(node)
                if edge.child is None:
                    child_board = node.board.copy(stack=True)
                    child_board.push(edge.move)
                    edge.child = _Node(child_board, f"node-{node_count}")
                    node_count += 1
                child = edge.child
                path.append((node, edge, child))
                node = child
                if not node.expanded:
                    break

            leaf, expansion = self._leaf(node, evaluator, root.board.turn)
            backups = self._backup(path, leaf.evaluation.value or 0.0)
            root_after = self._edge_stats(root, ValuePerspective.ROOT_PLAYER)
            events.append(
                SimulationEvent(
                    event_id=f"simulation-{simulation}",
                    simulation=simulation,
                    path=tuple(
                        PathStep(parent.node_id, edge.move.uci(), child.node_id) for parent, edge, child in path
                    ),
                    leaf=leaf,
                    backups=tuple(backups),
                    expansion=expansion,
                    root_before=root_before,
                    root_after=root_after,
                )
            )

        actions = tuple(RootAction(edge) for edge in self._edge_stats(root, ValuePerspective.ROOT_PLAYER))
        selection = self._root_selection(actions)
        return SearchTrace(
            root_fen=root.board.fen(),
            root_player=ChessPlayer.WHITE if root.board.turn else ChessPlayer.BLACK,
            capability=SearchCapability.REPLAYABLE,
            provenance=SearchProvenance(
                source="lczerolens-reference-mcts",
                engine="deterministic-reference",
                parameters=(
                    SearchParameter("c_puct", self.c_puct),
                    SearchParameter("selection", "Q + c_puct * P * sqrt(sum(N)) / (1 + N)"),
                    SearchParameter("tie_break", "UCI lexicographic order"),
                ),
            ),
            snapshots=(
                RootSnapshot(
                    sequence=0,
                    budget=SearchBudget(SearchBudgetUnit.SIMULATIONS, requested=simulations, observed=simulations),
                    evaluation=root_evaluation,
                    selection=selection,
                    actions=actions,
                ),
            ),
            events=tuple(events),
        )

    def _expand(
        self, node: _Node, evaluator: Evaluator | Callable[[LczeroBoard], TensorDict], perspective: ValuePerspective
    ) -> tuple[PositionEvaluation, NodeExpansion, EvaluatorCall]:
        output = self._single_evaluation(evaluator(node.board))
        policy = output.get("policy")
        value = output.get("value")
        if not isinstance(policy, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError("Evaluator must return TensorDict tensors named 'policy' and 'value'.")
        if value.numel() != 1 or not torch.isfinite(value).all() or not -1 <= value.item() <= 1:
            raise ValueError("Evaluator value must be one finite scalar in [-1, 1].")
        legal_moves = sorted(node.board.legal_moves, key=lambda move: move.uci())
        priors = node.board.get_legal_policy(policy.detach())
        legal_indices = node.board.get_legal_indices().tolist()
        by_index = {index: prior.item() for index, prior in zip(legal_indices, priors)}
        node.edges = {
            move.uci(): _Edge(move, by_index[node.board.encode_move(move, node.board.turn)]) for move in legal_moves
        }
        node.expanded = True
        edge_stats = self._edge_stats(node, perspective)
        legal_logits = policy.detach().gather(0, node.board.get_legal_indices().to(policy.device))
        by_index = {index: logit.item() for index, logit in zip(legal_indices, legal_logits)}
        evaluator_call = EvaluatorCall(
            dtype=str(policy.dtype).removeprefix("torch."),
            source_device=str(policy.device),
            search_device=str(policy.device),
            legal_policy_logits=tuple(
                (move.uci(), float(by_index[node.board.encode_move(move, node.board.turn)])) for move in legal_moves
            ),
        )
        return (
            PositionEvaluation(perspective, value=float(value.item())),
            NodeExpansion(node.node_id, edge_stats),
            evaluator_call,
        )

    @staticmethod
    def _single_evaluation(output: TensorDict) -> TensorDict:
        """Normalize the canonical evaluator's optional singleton batch."""
        if not isinstance(output, TensorDict):
            raise ValueError("Evaluator must return a TensorDict.")
        if output.batch_size == torch.Size([]):
            return output
        if output.batch_size == torch.Size([1]):
            return output[0]
        raise ValueError("Reference search requires one evaluator result per board.")

    def _leaf(
        self, node: _Node, evaluator: Evaluator | Callable[[LczeroBoard], TensorDict], root_turn: chess.Color
    ) -> tuple[LeafRecord, NodeExpansion | None]:
        perspective = ValuePerspective.ROOT_PLAYER if node.board.turn == root_turn else ValuePerspective.SIDE_TO_MOVE
        if node.board.is_game_over():
            outcome = node.board.outcome()
            value = (
                0.0
                if outcome is None or outcome.winner is None
                else 1.0
                if outcome.winner == node.board.turn
                else -1.0
            )
            return LeafRecord(node.node_id, PositionEvaluation(perspective, value=value), True), None
        evaluation, expansion, evaluator_call = self._expand(node, evaluator, perspective)
        return (
            LeafRecord(node.node_id, evaluation, False, evaluator_call),
            expansion,
        )

    def _select(self, node: _Node) -> _Edge:
        total_visits = sum(edge.visits for edge in node.edges.values())
        scale = math.sqrt(total_visits)
        scores = {
            move: edge.mean_value + self.c_puct * edge.prior * scale / (1 + edge.visits)
            for move, edge in node.edges.items()
        }
        best_score = max(scores.values())
        return node.edges[min(move for move, score in scores.items() if score == best_score)]

    def _backup(self, path: list[tuple[_Node, _Edge, _Node]], leaf_value: float) -> list[BackupUpdate]:
        value = leaf_value
        updates: list[BackupUpdate] = []
        for parent, edge, _ in reversed(path):
            value = -value
            perspective = ValuePerspective.ROOT_PLAYER if parent.node_id == "node-0" else ValuePerspective.SIDE_TO_MOVE
            before = self._edge_stat(edge, perspective)
            edge.visits += 1
            edge.total_value += value
            after = self._edge_stat(edge, perspective)
            updates.append(BackupUpdate(parent.node_id, value, before, after))
        return updates

    @staticmethod
    def _edge_stat(edge: _Edge, perspective: ValuePerspective) -> EdgeStatistics:
        return EdgeStatistics(edge.move.uci(), perspective, edge.prior, edge.visits, edge.total_value, edge.mean_value)

    def _edge_stats(self, node: _Node, perspective: ValuePerspective) -> tuple[EdgeStatistics, ...]:
        return tuple(self._edge_stat(node.edges[move], perspective) for move in sorted(node.edges))

    @staticmethod
    def _root_selection(actions: tuple[RootAction, ...]) -> RootSelection:
        best_visits = max(action.statistics.visits or 0 for action in actions)
        move = min(action.statistics.move for action in actions if (action.statistics.visits or 0) == best_visits)
        return RootSelection(move, rule="maximum visit count", tie_break="UCI lexicographic order", temperature=0.0)


def replay_root_events(events: tuple[SimulationEvent, ...]) -> tuple[EdgeStatistics, ...]:
    """Reconstruct final root state from events without accessing a search tree."""
    if not events:
        raise ValueError("At least one simulation event is required.")
    initial = events[0].root_before or ()
    state = {edge.move: edge for edge in initial}
    if len(state) != len(initial):
        raise ValueError("Replayable events require unique root moves.")
    if not state:
        raise ValueError("Replayable events require a non-empty root state.")
    for event in events:
        before = {edge.move: edge for edge in event.root_before or ()}
        after = {edge.move: edge for edge in event.root_after or ()}
        if len(before) != len(event.root_before or ()) or len(after) != len(event.root_after or ()):
            raise ValueError(f"Event {event.event_id} has duplicate root moves.")
        if before.keys() != after.keys():
            raise ValueError(f"Event {event.event_id} changes the root move set.")
        if before != state:
            raise ValueError(f"Event {event.event_id} does not chain from the reconstructed root state.")
        changed = [move for move in state if state[move] != after.get(move)]
        if len(changed) != 1:
            raise ValueError(f"Event {event.event_id} must change exactly one root edge.")
        move = changed[0]
        if not any(update.before == state[move] and update.after == after[move] for update in event.backups):
            raise ValueError(f"Event {event.event_id} has no matching root backup.")
        state = after
    return tuple(state[move] for move in sorted(state))


__all__ = ["Evaluator", "ReferenceMCTS", "replay_root_events"]
