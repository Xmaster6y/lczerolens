"""Contract tests for the stateless Lczero codecs."""

import chess
import pytest
import torch

from lczerolens._codec import InputFormat, decode_move, encode_input, encode_move, legal_indices, legal_mask


@pytest.mark.parametrize(
    ("input_format", "expected"),
    [
        (InputFormat.CLASSICAL_112, 32),
        (InputFormat.CLASSICAL_112_REPEATED, 32 * 8),
        (InputFormat.NO_HISTORY_REPEATED, 32 * 8),
        (InputFormat.NO_HISTORY_ZEROS, 32),
    ],
)
def test_initial_position_input_formats(input_format, expected):
    encoded = encode_input(chess.Board(), input_format=input_format)

    assert encoded.shape == (112, 8, 8)
    assert encoded.dtype is torch.float32
    assert encoded[:104].sum() == expected


def test_history_encoding_does_not_mutate_the_callers_board():
    board = chess.Board()
    for move in ("d2d4", "d7d5", "c2c4", "d5c4"):
        board.push_uci(move)
    fen = board.fen()
    history = tuple(board.move_stack)

    encoded = encode_input(board)

    assert encoded[:104].sum() == 31 + 32 * 4
    assert board.fen() == fen
    assert tuple(board.move_stack) == history


def test_move_vocabulary_round_trips_both_perspectives_and_promotions():
    white = chess.Board()
    white_move = chess.Move.from_uci("e2e4")
    assert decode_move(white, encode_move(white_move, white.turn)) == white_move

    black = chess.Board()
    black.push_uci("e2e4")
    black_move = chess.Move.from_uci("e7e5")
    assert decode_move(black, encode_move(black_move, black.turn)) == black_move

    for fen, uci in (
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
        ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1n"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
    ):
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        assert decode_move(board, encode_move(move, board.turn)) == move


def test_legal_indices_and_mask_share_the_fixed_policy_vocabulary():
    board = chess.Board()
    indices = legal_indices(board)
    mask = legal_mask(board)

    assert indices.shape == (20,)
    assert mask.shape == (1858,)
    assert mask.dtype is torch.bool
    assert mask.sum() == 20
    assert mask[indices].all()


@pytest.mark.parametrize(
    "fen",
    [
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
        "8/8/8/8/8/8/8/K6k w - - 0 1",
    ],
)
def test_terminal_position_has_no_semantic_legal_policy(fen):
    assert not legal_mask(chess.Board(fen)).any()


def test_codec_rejects_wrong_types_and_policy_indices():
    with pytest.raises(TypeError, match="chess.Board"):
        encode_input("not a board")
    with pytest.raises(TypeError, match="InputFormat"):
        encode_input(chess.Board(), input_format="classical")
    with pytest.raises(TypeError, match="chess.Move"):
        encode_move("e2e4", chess.WHITE)
    with pytest.raises(ValueError, match="Policy index"):
        decode_move(chess.Board(), 1858)
