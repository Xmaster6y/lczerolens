"""Search class"""

import chess
from lczerolens.board import LczeroBoard
import numpy as np
from tensordict import TensorDict
import torch
from typing import Dict, Protocol, Tuple, Optional


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
        n = len(board.legal_moves)
        return TensorDict(value=torch.Tensor([0.0]), policy=torch.full((n,), 1 / n))


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

        Returns
        -------
        None
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

    def search(
        self,
        root: Node,
        heuristic: Heuristic,
        iterations: int = 10,
    ) -> str:
        """Perform MCTS search on the given root node.

        Parameters
        ----------
            root : Node
                Node instance representing the current board state.
            heuristic : Heuristic
                Heuristic instance to evaluate board states.
            iterations : int
                Number of iterations to run the MCTS search.

        Returns
        -------
            best_move : str
                Best move as a string in UCI format.
        """
        if root.board.is_game_over():
            raise RuntimeError("Game already over.")

        if not root.initialized:
            root.set_evaluation(heuristic.evaluate(root))

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

        best_move = np.argmax(root.visits)
        return root.legal_moves[best_move]

    def _select_(
        self,
        node: Node,
    ) -> chess.Move:
        """Select the move to explore based on the PUCT formula.

        Parameters
        ----------
        node : Node
            Node instance representing the current board state.

        Returns
        -------
        move : chess.Move
            Selected move as a chess.Move object.
        """

        # PUCT formula = Q + U
        # Q = average value from simulations
        # U = exploration bonus encouraging less-visited moves
        Q = node.q_values
        U = self.c_puct * node.policy * (node.visits.sum() + 1) ** 0.5 / (1 + node.visits)
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
            if outcome.winner is not None:
                value = torch.Tensor([-1.0])
            else:
                value = torch.Tensor([0.0])
            td = TensorDict(value, None)
            node.set_evaluation(td)
        else:
            node.set_evaluation(heuristic.evaluate(node))
        return node.value

    @staticmethod
    def _backpropagate_(
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

    @staticmethod
    def plot(
        root: Node,
        max_depth: int = 3,
        filename: str = "mcts_tree",
    ) -> None:
        """Draw the MCTS tree using Graphviz and save it as a PNG.

        Parameters
        ----------
        root : Node
            Root node of the tree.
        max_depth : int
            Maximum depth to draw.
        filename : str
            Output PNG filename.

        Returns
        -------
        None
        """
        try:
            from graphviz import Digraph
        except ImportError as e:
            raise ImportError(
                "graphviz is required to render trees, install it with `pip install lczerolens[viz]`."
            ) from e

        dot = Digraph(comment="MCTS Tree")
        dot.attr("node", shape="circle")
        dot.node(str(id(root)), label=f"Root\nN={int(root.visits.sum().item())}")

        def add_nodes(
            node: Node,
            depth: int = 0,
        ) -> None:
            """Recursively add nodes to the graph.

            Parameters
            ----------
            node : Node
                Current node to add.
            depth : int
                Current depth in the tree.

            Returns
            -------
            None
            """

            if depth > max_depth:
                return

            for move, child in node.children.items():
                idx = node.legal_moves.index(move)
                n_visits = int(node.visits[idx].item())
                child_node = node.children[move]
                label = f"{move}\nN={n_visits}\nV={child_node.value[0]}"
                dot.node(str(id(child)), label=label)
                dot.edge(str(id(node)), str(id(child)))
                add_nodes(child, depth + 1)

        add_nodes(root, 0)
        dot.render(filename, format="png", cleanup=True)
