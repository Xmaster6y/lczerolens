"""
File to test the encodings for the Leela Chess Zero engine.
"""

import random

import chess
import pytest


@pytest.fixture(scope="module")
def random_move_board_list():
    board = chess.Board()
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
