"""
Wrapper tests.
"""

import chess
import torch

from lczerolens import board_utils


class TestTinySeNet:
    def test_senet_prediction(self, tiny_senet_ort, tiny_senet):
        """
        Test that the wrapper prediction works.
        """

        board = chess.Board()
        out = tiny_senet(board_utils.board_to_input_tensor(board).unsqueeze(0))
        policy = out["policy"]
        value = out["value"]
        onnx_policy, onnx_value = tiny_senet_ort.run(
            None,
            {
                "/input/planes": board_utils.board_to_input_tensor(board)
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
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            onnx_policy, onnx_value = tiny_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
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
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            onnx_policy, onnx_value = tiny_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
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
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            value = out["value"]
            onnx_policy, onnx_value = tiny_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_value = torch.tensor(onnx_value)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(value, onnx_value, atol=1e-4)


class TestMaiaSeNet:
    def test_senet_prediction(self, maia_senet_ort, maia_senet):
        """
        Test that the wrapper prediction works.
        """

        board = chess.Board()
        out = maia_senet(board_utils.board_to_input_tensor(board).unsqueeze(0))
        policy = out["policy"]
        wdl = out["wdl"]
        onnx_policy, onnx_wdl = maia_senet_ort.run(
            None,
            {
                "/input/planes": board_utils.board_to_input_tensor(board)
                .unsqueeze(0)
                .numpy()
            },
        )
        onnx_policy = torch.tensor(onnx_policy)
        onnx_wdl = torch.tensor(onnx_wdl)
        assert torch.allclose(policy, onnx_policy, atol=1e-4)
        assert torch.allclose(wdl, onnx_wdl, atol=1e-4)

    def test_senet_prediction_random(
        self, maia_senet_ort, maia_senet, random_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            out = maia_senet(
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            onnx_policy, onnx_wdl = maia_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_wdl = torch.tensor(onnx_wdl)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(wdl, onnx_wdl, atol=1e-4)

    def test_senet_prediction_repetition(
        self, maia_senet_ort, maia_senet, repetition_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            out = maia_senet(
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            onnx_policy, onnx_wdl = maia_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_wdl = torch.tensor(onnx_wdl)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(wdl, onnx_wdl, atol=1e-4)

    def test_senet_prediction_long(
        self, maia_senet_ort, maia_senet, long_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            out = maia_senet(
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            onnx_policy, onnx_wdl = maia_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
                    .unsqueeze(0)
                    .numpy()
                },
            )
            onnx_policy = torch.tensor(onnx_policy)
            onnx_wdl = torch.tensor(onnx_wdl)
            assert torch.allclose(policy, onnx_policy, atol=1e-4)
            assert torch.allclose(wdl, onnx_wdl, atol=1e-4)


class TestWinnerSeNet:
    def test_senet_prediction(self, winner_senet_ort, winner_senet):
        """
        Test that the wrapper prediction works.
        """

        board = chess.Board()
        out = winner_senet(
            board_utils.board_to_input_tensor(board).unsqueeze(0)
        )
        policy = out["policy"]
        wdl = out["wdl"]
        mlh = out["mlh"]
        onnx_policy, onnx_wdl, onnx_mlh = winner_senet_ort.run(
            None,
            {
                "/input/planes": board_utils.board_to_input_tensor(board)
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

    def test_senet_prediction_random(
        self, winner_senet_ort, winner_senet, random_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            out = winner_senet(
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            mlh = out["mlh"]
            onnx_policy, onnx_wdl, onnx_mlh = winner_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
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

    def test_senet_prediction_repetition(
        self, winner_senet_ort, winner_senet, repetition_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            out = winner_senet(
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            mlh = out["mlh"]
            onnx_policy, onnx_wdl, onnx_mlh = winner_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
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

    def test_senet_prediction_long(
        self, winner_senet_ort, winner_senet, long_move_board_list
    ):
        """
        Test that the wrapper prediction works.
        """

        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            out = winner_senet(
                board_utils.board_to_input_tensor(board).unsqueeze(0)
            )
            policy = out["policy"]
            wdl = out["wdl"]
            mlh = out["mlh"]
            onnx_policy, onnx_wdl, onnx_mlh = winner_senet_ort.run(
                None,
                {
                    "/input/planes": board_utils.board_to_input_tensor(board)
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
