"""
Class for wrapping the LCZero models.
"""

import os

import chess
import onnx
from onnx2pytorch import ConvertModel

from lczerolens import model_utils


class LczerroModelWrapper:
    """
    Class for wrapping the LCZero models.
    """

    def __init__(
        self,
        model_path: str,
    ):
        """
        Initializes the wrapper.
        """
        self.model_path = model_path
        self.model = None
        self.attention_cache = None
        self.num_attention_layers = None

    @classmethod
    def from_model(cls, model, model_path: str = ""):
        """
        Creates a wrapper from a model.
        """
        wrapper = cls(model_path)
        wrapper.model = model
        return wrapper

    def _load_model(self, model_path: str):
        """
        Loads the model.
        """
        if not os.path.exists(model_path):
            raise FileExistsError(f"Model path {model_path} does not exist.")
        try:
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
        except Exception:
            raise ValueError(f"Could not load model at {model_path}.")
        self.model = ConvertModel(onnx_model)

    def ensure_loaded(self):
        """
        Ensures that the model is loaded.
        """
        if self.model is None:
            self._load_model(self.model_path)
        if self.model is not None:
            self.model.eval()
        else:
            raise ValueError("Model is not loaded.")

    def prediction(self, board: chess.Board):
        """
        Predicts the move.
        """
        self.ensure_loaded()
        policy, outcome, value = model_utils.compute_move_prediction(
            self.model, [board]
        )
        return policy.squeeze(0), outcome.squeeze(0), value.squeeze(0)
