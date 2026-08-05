"""Fixture-backed conformance for the stateless Lczero codecs."""

import chess
import pytest
from lczero.backends import GameState

from lczerolens import backends as lczero_utils
from lczerolens._codec import encode_input, encode_move


@pytest.mark.backends
@pytest.mark.parametrize(
    "fixture_name",
    ("random_move_board_list", "repetition_move_board_list", "long_move_board_list"),
)
def test_input_encoding_matches_lczero_backend(fixture_name, request, tiny_lczero_backend):
    move_list, board_list = request.getfixturevalue(fixture_name)
    for index, board in enumerate(board_list):
        game = GameState(moves=[move.uci() for move in move_list[:index]])
        expected = lczero_utils.board_from_backend(tiny_lczero_backend, game)
        assert (encode_input(board) == expected).all()


@pytest.mark.backends
@pytest.mark.parametrize(
    "backend_moves",
    (["e1h1", "e1a1"], ["e1g1", "e1c1"]),
)
def test_castling_policy_indices_are_canonical(backend_moves):
    class Game:
        def moves(self):
            return backend_moves

        def policy_indices(self):
            return [103, 97]

    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    moves, indices = lczero_utils.moves_with_castling_swap(Game(), board)

    assert moves == ["e1g1", "e1c1"]
    assert indices == [102, 99]


@pytest.mark.backends
@pytest.mark.parametrize("fixture_name", ("random_move_board_list", "long_move_board_list"))
def test_policy_vocabulary_matches_lczero_backend(fixture_name, request):
    move_list, board_list = request.getfixturevalue(fixture_name)
    for index, board in enumerate(board_list):
        game = GameState(moves=[move.uci() for move in move_list[:index]])
        backend_moves, backend_indices = lczero_utils.moves_with_castling_swap(game, board)
        expected_moves = [move.uci() for move in board.legal_moves]
        expected_indices = [encode_move(board, move) for move in board.legal_moves]
        assert set(backend_moves) == set(expected_moves)
        assert set(backend_indices) == set(expected_indices)
