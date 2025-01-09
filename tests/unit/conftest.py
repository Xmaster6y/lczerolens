"""
File to test the encodings for the Leela Chess Zero engine.
"""

import random
import chess
import pytest

from lczerolens import LczeroBoard


@pytest.fixture(scope="module")
def random_move_board_list():
    board = LczeroBoard()
    seed = 42
    random.seed(seed)
    move_list = []
    board_list = [board.copy()]
    for _ in range(20):
        move = random.choice(list(board.legal_moves))
        move_list.append(move)
        board.push(move)
        board_list.append(board.copy(stack=8))
    return move_list, board_list


@pytest.fixture(scope="module")
def repetition_move_board_list():
    board = LczeroBoard()
    move_list = []
    board_list = [board.copy()]
    for uci_move in ("b1a3", "b8c6", "a3b1", "c6b8") * 4:
        move = chess.Move.from_uci(uci_move)
        move_list.append(move)
        board.push(move)
        board_list.append(board.copy(stack=True))  # Full stack is needed for repetition detection
    return move_list, board_list


@pytest.fixture(scope="module")
def long_move_board_list():
    board = LczeroBoard()
    seed = 6
    random.seed(seed)
    move_list = []
    board_list = [board.copy()]
    for _ in range(80):
        move = random.choice(list(board.legal_moves))
        move_list.append(move)
        board.push(move)
        board_list.append(board.copy(stack=8))
    return move_list, board_list
