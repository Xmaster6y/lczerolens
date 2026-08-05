"""Model tests."""

import pytest
import torch
import chess
from lczero.backends import GameState
from huggingface_hub import delete_repo
from torch import nn
from tensordict import TensorDict

from lczerolens._codec import encode_input
from lczerolens.model import PolicyFlow, ValueFlow, WdlFlow, MlhFlow, ForceValue, LczeroModel
from lczerolens import backends as lczero_utils


def _model_input(board: chess.Board) -> TensorDict:
    planes = encode_input(board).unsqueeze(0)
    return TensorDict({"board": planes}, batch_size=[1])


@pytest.mark.backends
class TestModel:
    def test_model_prediction(self, tiny_lczero_backend, tiny_model):
        """Test that the model prediction works."""
        board = chess.Board()
        (out,) = tiny_model(_model_input(board))
        policy = out["policy"]
        value = out["value"]
        lczero_game = GameState()
        lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
        assert torch.allclose(policy, lczero_policy, atol=1e-4)
        assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_model_prediction_random(self, tiny_lczero_backend, tiny_model, random_move_board_list):
        """Test that the model prediction works."""
        move_list, board_list = random_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_model(_model_input(board))
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_model_prediction_repetition(self, tiny_lczero_backend, tiny_model, repetition_move_board_list):
        """Test that the model prediction works."""
        move_list, board_list = repetition_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_model(_model_input(board))
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)

    def test_model_prediction_long(self, tiny_lczero_backend, tiny_model, long_move_board_list):
        """Test that the model prediction works."""
        move_list, board_list = long_move_board_list
        for i, board in enumerate(board_list):
            (out,) = tiny_model(_model_input(board))
            policy = out["policy"]
            value = out["value"]
            lczero_game = GameState(moves=[move.uci() for move in move_list[:i]])
            lczero_policy, lczero_value = lczero_utils.prediction_from_backend(tiny_lczero_backend, lczero_game)
            assert torch.allclose(policy, lczero_policy, atol=1e-4)
            assert torch.allclose(value, lczero_value, atol=1e-4)


class TestManageModels:
    def test_tensordict_input_remains_transparent(self):
        model = LczeroModel(nn.Identity(), out_keys=["output"])
        inputs = TensorDict({"board": torch.zeros((1, 112, 8, 8))}, batch_size=1)
        output = model(inputs)
        assert output["output"].shape == (1, 112, 8, 8)

    def test_from_model_requires_explicit_keys_for_plain_modules(self):
        with pytest.raises(ValueError, match="explicit out_keys"):
            LczeroModel.from_model(nn.Identity())

    def test_onnx_loading_without_shape_check_passes_the_path(self, monkeypatch, tmp_path):
        path = tmp_path / "model.onnx"
        path.touch()
        converted = nn.Identity()
        monkeypatch.setattr("lczerolens.model.convert", lambda value: converted)
        monkeypatch.setattr(LczeroModel, "from_model", classmethod(lambda cls, model, **_: model))
        assert LczeroModel.from_onnx_path(str(path), check=False) is converted

    def test_model_from_hf_is_hermetic(self, monkeypatch):
        """The Hub adapter forwards its download result without contacting the network."""
        import huggingface_hub

        downloaded_path = "/tmp/downloaded-model.pt"
        expected_model = object()
        download_calls = []

        def fake_download(repo_id, filename, **kwargs):
            download_calls.append((repo_id, filename, kwargs))
            return downloaded_path

        def fake_from_path(cls, path, **kwargs):
            assert cls is LczeroModel
            assert path == downloaded_path
            assert kwargs == {"weights_only": True}
            return expected_model

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
        monkeypatch.setattr(LczeroModel, "from_path", classmethod(fake_from_path))

        assert (
            LczeroModel.from_hf("owner/model", "weights.pt", {"revision": "abc"}, weights_only=True) is expected_model
        )
        assert download_calls == [("owner/model", "weights.pt", {"revision": "abc"})]

    def test_model_push_to_hf_is_hermetic(self, monkeypatch):
        """The Hub adapter creates, serializes, and uploads without a live repository."""
        import huggingface_hub

        created = []
        uploaded = []
        monkeypatch.setattr(huggingface_hub, "repo_exists", lambda repo_id, token: False)
        monkeypatch.setattr(
            huggingface_hub, "create_repo", lambda repo_id, **kwargs: created.append((repo_id, kwargs))
        )
        monkeypatch.setattr(
            huggingface_hub,
            "upload_file",
            lambda path_or_fileobj, repo_id, path_in_repo, **kwargs: uploaded.append(
                (path_or_fileobj, repo_id, path_in_repo, kwargs)
            ),
        )

        model = LczeroModel(nn.Identity(), out_keys=["output"])
        model.push_to_hf(
            "owner/model",
            create_kwargs={"token": "secret", "private": True},
            path_in_repo="weights/model.pt",
            commit_message="test upload",
        )

        assert created == [("owner/model", {"token": "secret", "private": True})]
        assert len(uploaded) == 1
        _, repo_id, path_in_repo, upload_kwargs = uploaded[0]
        assert (repo_id, path_in_repo) == ("owner/model", "weights/model.pt")
        assert upload_kwargs == {"commit_message": "test upload"}

    @pytest.mark.network
    def test_model_from_hf(self):
        """Test that the model save and load works."""
        board = chess.Board()
        model = LczeroModel.from_hf("lczerolens/maia-1100")
        output = model(_model_input(board))
        assert "policy" in output
        assert "wdl" in output

    @pytest.mark.network
    def test_model_push_to_hf(self):
        """Test that the model push to hf works."""
        board = chess.Board()
        model = LczeroModel.from_hf("lczerolens/maia-1100")
        model.push_to_hf("lczerolens/tests")
        output = model(_model_input(board))
        assert "policy" in output
        assert "wdl" in output
        delete_repo("lczerolens/tests")


class TestFlows:
    def test_policy_flow(self, tiny_model):
        """Test that the policy flow works."""
        policy_flow = PolicyFlow.from_model(tiny_model.module)
        board = chess.Board()
        output = policy_flow(_model_input(board))
        assert "value" not in output

    def test_value_flow(self, tiny_model):
        """Test that the value flow works."""
        value_flow = ValueFlow.from_model(tiny_model.module)
        board = chess.Board()
        output = value_flow(_model_input(board))
        assert "policy" not in output

    def test_wdl_flow(self, winner_model):
        """Test that the wdl flow works."""
        wdl_flow = WdlFlow.from_model(winner_model.module)
        board = chess.Board()
        output = wdl_flow(_model_input(board))
        assert "policy" not in output

    def test_mlh_flow(self, winner_model):
        """Test that the mlh flow works."""
        mlh_flow = MlhFlow.from_model(winner_model.module)
        board = chess.Board()
        output = mlh_flow(_model_input(board))
        assert "policy" not in output

    def test_force_value(self, tiny_model):
        """Test that the force value flow works."""
        force_value = ForceValue.from_model(tiny_model.module)
        board = chess.Board()
        output = force_value(_model_input(board))
        assert "value" in output

    def test_force_value_wdl(self, winner_model):
        """Test that the force value flow works."""
        force_value = ForceValue.from_model(winner_model.module)
        board = chess.Board()
        output = force_value(_model_input(board))
        assert "value" in output

        value = output["wdl"] @ torch.tensor([1.0, 0.0, -1.0], device=output.device)
        assert torch.allclose(output["value"], value)

    def test_incompatible_flows(self, tiny_model, winner_model):
        """Test that the flows raise an error *
        when the model is incompatible.
        """
        with pytest.raises(ValueError):
            ValueFlow.from_model(winner_model)
        with pytest.raises(ValueError):
            WdlFlow.from_model(tiny_model)
        with pytest.raises(ValueError):
            MlhFlow.from_model(tiny_model)
