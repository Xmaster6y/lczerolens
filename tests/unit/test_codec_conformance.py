"""Fixture-backed conformance for the stateless Lczero codecs."""

import chess
import pytest
import torch

from lczerolens._codec import encode_input, encode_move

GameState = pytest.importorskip(
    "lczero.backends", reason="native Lczero bindings are a conformance-only dependency"
).GameState


def _input_from_backend(backend, game):
    """Materialize the native binding input for codec conformance only."""
    backend_input = game.as_input(backend)
    planes = torch.zeros((112, 64), dtype=torch.float)
    for plane in range(112):
        mask = f"{backend_input.mask(plane):b}".zfill(64)
        planes[plane] = torch.tensor(tuple(map(int, reversed(mask))), dtype=torch.float) * backend_input.val(plane)
    return planes.view((112, 8, 8))


def _moves_with_castling_swap(game, board):
    """Normalize native binding castling moves for vocabulary conformance."""
    backend_moves = list(game.moves())
    backend_indices = list(game.policy_indices())
    for move in board.legal_moves:
        if not board.is_castling(move):
            continue
        rook_file = 7 if board.is_kingside_castling(move) else 0
        rook_square = chess.square(rook_file, chess.square_rank(move.from_square))
        accepted_moves = (move.uci(), chess.Move(move.from_square, rook_square).uci())
        try:
            index = next(i for i, backend_move in enumerate(backend_moves) if backend_move in accepted_moves)
        except StopIteration:
            continue
        backend_moves[index] = move.uci()
    return backend_moves, backend_indices


@pytest.mark.conformance
@pytest.mark.parametrize(
    "fixture_name",
    ("random_move_board_list", "repetition_move_board_list", "long_move_board_list"),
)
def test_input_encoding_matches_lczero_backend(fixture_name, request, tiny_lczero_backend):
    move_list, board_list = request.getfixturevalue(fixture_name)
    for index, board in enumerate(board_list):
        game = GameState(moves=[move.uci() for move in move_list[:index]])
        expected = _input_from_backend(tiny_lczero_backend, game)
        assert (encode_input(board) == expected).all()


@pytest.mark.conformance
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
    moves, indices = _moves_with_castling_swap(Game(), board)

    assert moves == ["e1g1", "e1c1"]
    assert indices == [103, 97]


@pytest.mark.conformance
@pytest.mark.parametrize("fixture_name", ("random_move_board_list", "long_move_board_list"))
def test_policy_vocabulary_matches_lczero_backend(fixture_name, request):
    move_list, board_list = request.getfixturevalue(fixture_name)
    for index, board in enumerate(board_list):
        game = GameState(moves=[move.uci() for move in move_list[:index]])
        backend_moves, backend_indices = _moves_with_castling_swap(game, board)
        expected_moves = [move.uci() for move in board.legal_moves]
        expected_indices = [encode_move(board, move) for move in board.legal_moves]
        assert set(backend_moves) == set(expected_moves)
        assert set(backend_indices) == set(expected_indices)
