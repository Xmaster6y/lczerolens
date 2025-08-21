"""
File to test the mcts module of lczerolens.
"""

import chess
import numpy as np
import pytest
import torch
from lczerolens import LczeroBoard
from tensordict import TensorDict
from src.lczerolens.search import MCTS, Node, DummyHeuristic


def test_node_initialization():
    board = LczeroBoard()
    node = Node(board, None)

    assert node.parent is None
    assert not node.is_terminal

    assert len(node.children) == 0
    assert len(node.legal_moves) == 20
    assert torch.all(node.visits == 0)
    assert torch.all(node.q_values == 0)

    assert node.value is None
    assert node.policy is None
    assert not node.initialized


def test_node_terminal():
    board = LczeroBoard("8/8/8/8/8/8/8/K6k w - - 0 1")
    node = Node(board, None)

    assert node.is_terminal


def test_set_evaluation_once():
    board = LczeroBoard()
    node = Node(board, None)

    td = TensorDict({"value": torch.tensor([5.0]), "policy": torch.ones(len(node.legal_moves))})
    node.set_evaluation(td)

    assert torch.equal(node.value, torch.tensor([5.0]))
    assert torch.equal(node.policy, torch.ones(len(node.legal_moves)))

    with pytest.raises(RuntimeError):
        node.set_evaluation(td)


def test_dummy_heuristic_output():
    board = LczeroBoard()
    heuristic = DummyHeuristic()
    td = heuristic.evaluate(board)

    assert "value" in td.keys()
    assert "policy" in td.keys()

    assert td.get("value").item() == 0.0
    assert td.get("policy").shape[0] == board.legal_moves.count()
    assert abs(td.get("policy").sum().item() - 1.0) < 1e-6


def test_mcts_search_best_move():
    print("1\n")
    board = LczeroBoard("2K5/k7/8/8/8/8/8/1Q6 w - - 0 1")
    heuristic = DummyHeuristic()
    mcts = MCTS(c_puct=10)
    root = Node(board, None)
    mcts.search_(root, heuristic, iterations=100)
    best_move = np.argmax(root.visits)

    assert root.legal_moves[best_move].uci() == "b1b7"


def test_mcts_search_terminal_root():
    board = LczeroBoard("2K5/kQ6/8/8/8/8/8/8 b - - 0 1")
    node = Node(board, None)
    mcts = MCTS()
    heuristic = DummyHeuristic()

    try:
        mcts.search_(node, heuristic=heuristic, iterations=10)
    except RuntimeError as e:
        assert str(e) == "Game already over."
    else:
        assert False, "Expected RuntimeError for terminal root"


def test_mcts_select_returns_legal_move():
    board = LczeroBoard()
    root = Node(board, None)
    heuristic = DummyHeuristic()
    mcts = MCTS()

    mcts._evaluate_(root, heuristic)
    move = mcts._select_(root)

    assert move in root.legal_moves


def test_mcts_evaluate_terminal():
    board = LczeroBoard("2K5/kQ6/8/8/8/8/8/8 b - - 0 1")
    heuristic = DummyHeuristic()
    mcts = MCTS()
    node = Node(board, None)
    value_tensor = mcts._evaluate_(node, heuristic)

    assert value_tensor.item() == -1.0


def test_mcts_evaluate_non_terminal():
    board = LczeroBoard("kq6/8/8/8/8/8/8/7K w - - 0 1")
    heuristic = DummyHeuristic()
    mcts = MCTS()
    node = Node(board, None)
    value_tensor = mcts._evaluate_(node, heuristic)

    assert value_tensor.item() == 0.0


def test_mcts_backpropagate_updates_q_values():
    board = LczeroBoard("2K5/k7/8/8/8/8/8/1Q6 w - - 0 1")
    mcts = MCTS()
    root = Node(board, None)
    move = chess.Move.from_uci("b1b7")
    child_board = root.board.copy()
    child_board.push(move)
    child = Node(child_board, root)
    child._value = torch.Tensor([1.0])
    child._initialized = True
    root.children[move] = child
    mcts._backpropagate_(child, child.value.item())
    move_index = root.legal_moves.index(move)

    assert root.q_values[move_index] == -1.0


# def test_plot_creates_file(tmp_path):
#     board = LczeroBoard()
#     root = Node(board, None)
#     filename = tmp_path / "tree"
#     MCTS.plot(root, max_depth=1, filename=str(filename))

#     assert os.path.exists(str(filename) + ".png")
