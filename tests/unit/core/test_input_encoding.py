"""
Tests for the input encoding.
"""

import pytest
import chess

from lczerolens import LczeroBoard, InputEncoding


class TestInputEncoding:
    @pytest.mark.parametrize(
        "input_encoding_expected",
        [
            (InputEncoding.INPUT_CLASSICAL_112_PLANE, 32),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED, 32 * 8),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY_REPEATED, 32 * 8),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY_ZEROS, 32),
        ],
    )
    def test_sum_initial_config_planes(self, input_encoding_expected):
        """
        Test the sum of the config planes for the initial board.
        """
        input_encoding, expected = input_encoding_expected
        board = LczeroBoard()
        board_tensor = board.to_input_tensor(input_encoding=input_encoding)
        assert board_tensor[:104].sum() == expected

    @pytest.mark.parametrize(
        "input_encoding_expected",
        [
            (InputEncoding.INPUT_CLASSICAL_112_PLANE, 31 + 32 * 4),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_REPEATED, 31 + 32 * 7),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY_REPEATED, 31 * 8),
            (InputEncoding.INPUT_CLASSICAL_112_PLANE_NO_HISTORY_ZEROS, 31),
        ],
    )
    def test_sum_qga_config_planes(self, input_encoding_expected):
        """
        Test the sum of the config planes for the queen's gambit accepted.
        """
        input_encoding, expected = input_encoding_expected
        board = LczeroBoard()
        moves = [
            chess.Move.from_uci("d2d4"),
            chess.Move.from_uci("d7d5"),
            chess.Move.from_uci("c2c4"),
            chess.Move.from_uci("d5c4"),
        ]
        for move in moves:
            board.push(move)
        board_tensor = board.to_input_tensor(input_encoding=input_encoding)
        assert board_tensor[:104].sum() == expected
