"""
Wrapper tests.
"""

import chess
import torch

from lczerolens import board_utils


class TestSeNet:
    def test_senet_prediction(self, tiny_senet_ort, tiny_senet):
        """
        Test that the wrapper prediction works.
        """

        board = chess.Board()
        out = tiny_senet(
            board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
        )
        policy = out["policy"]
        value = out["value"]
        onnx_policy, onnx_value = tiny_senet_ort.run(
            None,
            {
                "/input/planes": board_utils.board_to_tensor112x8x8(board)
                .unsqueeze(0)
                .numpy()
            },
        )
        onnx_policy = torch.tensor(onnx_policy)
        onnx_value = torch.tensor(onnx_value)
        assert torch.allclose(policy, onnx_policy, atol=1e-4)
        assert torch.allclose(value, onnx_value, atol=1e-4)

    def test_senet_prediction_random(
        self, tiny_senet_ort, tiny_senet, random_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_senet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            onnx_policy, onnx_value = tiny_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_tensor112x8x8(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_value = torch.tensor(onnx_value)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(value, onnx_value, atol=1e-4)

    def test_senet_prediction_repetition(
        self, tiny_senet_ort, tiny_senet, repetition_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_senet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            onnx_policy, onnx_value = tiny_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_tensor112x8x8(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_value = torch.tensor(onnx_value)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(value, onnx_value, atol=1e-4)

    def test_senet_prediction_long(
        self, tiny_senet_ort, tiny_senet, long_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            out = tiny_senet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            onnx_policy, onnx_value = tiny_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_tensor112x8x8(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_value = torch.tensor(onnx_value)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(value, onnx_value, atol=1e-4)
