"""Search class"""

import chess
import numpy as np
import torch
from lczerolens.board import LczeroBoard
from lczerolens.model import ForceValue, LczeroModel
from tensordict import TensorDict
from typing import Callable, Dict, Optional, Protocol, Tuple


class Heuristic(Protocol):
    """Heuristic protocol for evaluating chess positions."""

    def evaluate(
        self,
        board: LczeroBoard,
    ) -> TensorDict:
        """
        Evaluate a single board.

        Parameters
        ----------
        board : LczeroBoard
            LczeroBoard instance representing the current board state

        Returns
        -------
        TensorDict
            Dictionary with fileds value and policy.
        """
        ...


class DummyHeuristic:
    """Simple heuristic for MCTS."""

    def evaluate(
        self,
        board: LczeroBoard,
    ) -> TensorDict:
        """Evaluate a single board.

        Parameters
        ----------
        board : LczeroBoard
            LczeroBoard instance

        Returns
        -------
        TensorDict
            Dictionary with fileds value and policy.
        """
        n = board.legal_moves.count()
        return TensorDict(value=torch.Tensor([0.0]), policy=torch.full((n,), 1 / n))


class MaterialHeuristic:
    """Heuristic 'model' that outputs uniform policy and material advantage as value."""

    piece_values_default = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 100,
    }

    def __init__(
        self,
        piece_values: Optional[Dict[int, int]] = None,
        normalization_constant: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ):
        self.piece_values = piece_values or self.piece_values_default
        self.normalization_constant = normalization_constant
        self.activation = activation

    def evaluate(
        self,
        board: LczeroBoard,
    ) -> TensorDict:
        """
        Compute the label for a given model and input.
        """
        us, them = board.turn, not board.turn
        relative_value = 0
        for piece in range(1, 7):
            relative_value += len(board.pieces(piece, us)) * self.piece_values[piece]
            relative_value -= len(board.pieces(piece, them)) * self.piece_values[piece]

        value = self.activation(torch.tensor([relative_value / self.normalization_constant], dtype=torch.float32))

        n = board.legal_moves.count()
        policy = torch.full((n,), 1 / n)
        return TensorDict(value=value, policy=policy)


class ModelHeuristic:
    """Evaluate boards using a neural network model for MCTS."""

    def __init__(self, model: LczeroModel):
        self._model = ForceValue.from_model(model.module)

    def evaluate(self, board: LczeroBoard) -> TensorDict:
        """
        Evaluate a single board using the Lczero model.

        Returns TensorDict with 'value' and 'policy'.
        """
        td = self._model(board)[0]
        legal_indices = board.get_legal_indices()
        td["policy"] = td["policy"].gather(0, legal_indices)
        return td


class Node:
    """Node for MCTS using LczeroBoard."""

    def __init__(
        self,
        board: LczeroBoard,
        parent: "Node",
    ) -> None:
        """Initialize a Node with a given board and parent node."""
        self.board = board
        self.parent = parent
        self.is_terminal: bool = board.is_game_over()

        self.children: Dict[chess.Move, "Node"] = {}
        self.legal_moves: Tuple[chess.Move, ...] = tuple(board.legal_moves)
        self.visits: torch.Tensor = torch.zeros(len(self.legal_moves))
        self.q_values: torch.Tensor = torch.zeros(len(self.legal_moves))

        self._value: Optional[torch.Tensor] = None
        self._policy: Optional[torch.Tensor] = None
        self._initialized: bool = False

    @property
    def value(self):
        return self._value

    @property
    def policy(self):
        return self._policy

    @property
    def initialized(self):
        return self._initialized

    def set_evaluation(
        self,
        td: TensorDict,
    ) -> None:
        """Set the evaluation for the node.

        Parameters
        ----------
        td : TensorDict
            TensorDict containing value and policy tensors.
        """
        if self._value is not None or self._policy is not None:
            raise RuntimeError("Node already initialized.")

        self._value = td.get("value")
        self._policy = td.get("policy")
        self._initialized = True


class MCTS:
    """Monte Carlo Tree Search with PUCT formula."""

    def __init__(
        self,
        c_puct: float = 1.0,
        n_parallel_rollouts: int = 1,
    ):
        """Initialize the class."""
        self.c_puct = c_puct
        self.n_parallel_rollouts = n_parallel_rollouts

    def search_(
        self,
        root: Node,
        heuristic: Heuristic,
        iterations: int = 10,
    ) -> None:
        """Perform MCTS search on the given root node.

        Parameters
        ----------
            root : Node
                Node instance representing the current board state.
            heuristic : Heuristic
                Heuristic instance to evaluate board states.
            iterations : int
                Number of iterations to run the MCTS search.
        """
        if root.board.is_game_over():
            raise RuntimeError("Game already over.")

        if not root.initialized:
            root.set_evaluation(heuristic.evaluate(root.board))

        for _ in range(iterations):
            node = root
            done = False

            # Selection
            while not done:
                move = self._select_(node)

                # Expansion
                if move not in node.children:
                    done = True
                    new_board = node.board.copy()
                    new_board.push(move)
                    node.children[move] = Node(board=new_board, parent=node)

                node = node.children[move]
                done = done or node.is_terminal

            # Evaluation
            value = self._evaluate_(node, heuristic)

            # Backpropagation
            self._backpropagate_(node, value)

    def _select_(
        self,
        node: Node,
    ) -> chess.Move:
        """Select the move to explore based on the PUCT formula."""

        Q = node.q_values.detach() if node.q_values.requires_grad else node.q_values
        policy = node.policy.detach() if node.policy.requires_grad else node.policy

        # PUCT formula = Q + U
        # Q = average value from simulations
        # U = exploration bonus encouraging less-visited moves
        U = self.c_puct * policy * ((node.visits.sum() + 1) ** 0.5) / (1 + node.visits)

        if isinstance(Q, torch.Tensor) and isinstance(U, torch.Tensor):
            a = torch.argmax(Q + U).item()
        else:
            a = np.argmax(Q + U)

        node.visits[a] += 1
        return node.legal_moves[a]

    def _evaluate_(
        self,
        node: Node,
        heuristic: Heuristic,
    ) -> torch.Tensor:
        """Evaluate a single board.

        Parameters
        ----------
        node : Node
            Node instance representing the current board state.
        heuristic : Heuristic
            Heuristic instance to evaluate board states.

        Returns
        -------
        value : torch.Tensor
            Value tensor for the current node.
        """
        if node.initialized:
            return node.value
        elif node.is_terminal:
            outcome = node.board.outcome()
            value = torch.Tensor([0.0]) if outcome.winner is None else torch.Tensor([-1.0])
            td = TensorDict(value=value, policy=None)
            node.set_evaluation(td)
        else:
            node.set_evaluation(heuristic.evaluate(node.board))
        return node.value

    def _backpropagate_(
        self,
        node: Node,
        value: float,
    ) -> None:
        """Backpropagate the reward from the leaf node to the root node.

        Parameters
        ----------
        node : Node
            Node instance representing the leaf node.
        value : float
            Float value to backpropagate.
        """
        while node.parent is not None:
            parent = node.parent
            value = -value
            move = node.board.move_stack[-1]
            idx = parent.legal_moves.index(move)
            parent.q_values[idx] = (parent.q_values[idx] * parent.visits[idx] + value) / (parent.visits[idx] + 1)
            node = parent

    def render_tree(
        root: Node, max_depth: int = 3, save_to: Optional[str] = None, min_visit_percentage: float = 0.0
    ) -> Optional[str]:
        """
        Render the MCTS tree as an SVG.

        Parameters
        ----------
        root : Node
            Root node of the tree.
        max_depth : int, default=3
            Maximum depth to render.
        save_to : Optional[str], default=None
            Path to save the SVG. If None, returns the SVG string.

        Returns
        -------
        Optional[str]
            SVG string of the tree, or None if saved to file.
        """
        try:
            from graphviz import Digraph
        except ImportError as e:
            raise ImportError(
                "graphviz is required to render trees, install it with `pip install lczerolens[viz]`."
            ) from e

        dot = Digraph(comment="MCTS Tree")
        dot.attr("graph", rankdir="TB", ranksep="1.5")
        dot.attr("node", shape="circle")
        dot.node(str(id(root)), label=f"Root\nN={int(root.visits.sum().item())}")

        def add_nodes(node: Node, depth: int = 0):
            if depth > max_depth:
                return
            if not node.children:
                return
            visit_percentages = node.visits / node.visits.sum()
            for move, child in node.children.items():
                child_index = node.legal_moves.index(move)
                if visit_percentages[child_index] < min_visit_percentage:
                    continue
                idx = node.legal_moves.index(move)
                n_visits = int(node.visits[idx].item())
                label = f"{move}\nN={n_visits}\nV={child.value.item():.3f}"
                dot.node(str(id(child)), label=label)
                dot.edge(str(id(node)), str(id(child)))
                add_nodes(child, depth + 1)

        add_nodes(root, 0)

        svg_tree = dot.pipe(format="svg").decode("utf-8")

        if save_to is not None:
            if not save_to.endswith(".svg"):
                raise ValueError("Only saving to `.svg` is supported")
            with open(save_to, "w") as f:
                f.write(svg_tree)
            return None

        return svg_tree
