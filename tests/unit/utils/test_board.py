"""
Tests for the board utils.
"""

import torch

from lczerolens import board_utils


class TestEncoding:
    def test_board_to_tensor13x8x8(self, random_move_board_list):
        """
        Test that the board to tensor function works.
        """
        _, board_list = random_move_board_list
        for board in board_list:
            board_tensor = board_utils.board_to_tensor13x8x8(board)
            planes = board_tensor.sum(dim=(1, 2))
            assert planes.shape == (13,)
            assert (
                planes[:12]
                == torch.tensor([8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1])
            ).all()
            assert planes[12] == 0

    def test_board_to_tensor112x8x8(self, random_move_board_list):
        """
        Test that the board to tensor function works.
        """
        _, board_list = random_move_board_list
        for board in board_list:
            board_tensor = board_utils.board_to_tensor112x8x8(board)
            planes = board_tensor.sum(dim=(1, 2))
            assert planes.shape == (112,)
            previous_steps = min(len(board.move_stack) + 1, 8)
            for i in range(previous_steps):
                assert (
                    planes[13 * i : 13 * (i + 1) - 1]
                    == torch.tensor([8, 2, 2, 2, 1, 1, 8, 2, 2, 2, 1, 1])
                ).all()
            if previous_steps < 8:
                assert planes[13 * previous_steps : 13 * 8].sum() == 0
