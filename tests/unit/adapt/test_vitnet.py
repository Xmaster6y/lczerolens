"""
Wrapper tests.
"""

import chess
import pytest
import torch

from lczerolens import board_utils


class TestT1VitNet:
    @pytest.mark.xfail(reason="T1VitNet not yet implemented")
    def test_vitnet_prediction(self, t1_vitnet_ort, t1_vitnet):
        """
        Test that the wrapper prediction works.
        """

        board = chess.Board()
        out = t1_vitnet(board_utils.board_to_tensor112x8x8(board).unsqueeze(0))
        policy = out["policy"]
        wdl = out["wdl"]
        mlh = out["mlh"]
        onnx_policy, onnx_wdl, onnx_mlh = t1_vitnet_ort.run(
            None,
            {
                "/input/planes": board_utils.board_to_tensor112x8x8(board)
                .unsqueeze(0)
                .numpy()
            },
        )
        onnx_policy = torch.tensor(onnx_policy)
        onnx_wdl = torch.tensor(onnx_wdl)
        onnx_mlh = torch.tensor(onnx_mlh)
        assert torch.allclose(policy, onnx_policy, atol=1e-4)
        assert torch.allclose(wdl, onnx_wdl, atol=1e-4)
        assert torch.allclose(mlh, onnx_mlh, atol=1e-4)

    @pytest.mark.xfail(reason="T1VitNet not yet implemented")
    def test_vitnet_prediction_random(
        self, t1_vitnet_ort, t1_vitnet, random_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            out = t1_vitnet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            mlh = out["mlh"]
            onnx_policy, onnx_wdl, onnx_mlh = t1_vitnet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_tensor112x8x8(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_wdl = torch.tensor(onnx_wdl)
            onnx_mlh = torch.tensor(onnx_mlh)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(wdl, onnx_wdl, atol=1e-4)
            assert torch.allclose(mlh, onnx_mlh, atol=1e-4)

    @pytest.mark.xfail(reason="T1VitNet not yet implemented")
    def test_vitnet_prediction_repetition(
        self, t1_vitnet_ort, t1_vitnet, repetition_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            out = t1_vitnet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            mlh = out["mlh"]
            onnx_policy, onnx_wdl, onnx_mlh = t1_vitnet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_tensor112x8x8(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_wdl = torch.tensor(onnx_wdl)
            onnx_mlh = torch.tensor(onnx_mlh)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(wdl, onnx_wdl, atol=1e-4)
            assert torch.allclose(mlh, onnx_mlh, atol=1e-4)

    @pytest.mark.xfail(reason="T1VitNet not yet implemented")
    def test_vitnet_prediction_long(
        self, t1_vitnet_ort, t1_vitnet, long_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            out = t1_vitnet(
                board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            mlh = out["mlh"]
            onnx_policy, onnx_wdl, onnx_mlh = t1_vitnet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_tensor112x8x8(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_wdl = torch.tensor(onnx_wdl)
            onnx_mlh = torch.tensor(onnx_mlh)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(wdl, onnx_wdl, atol=1e-4)
            assert torch.allclose(mlh, onnx_mlh, atol=1e-4)
