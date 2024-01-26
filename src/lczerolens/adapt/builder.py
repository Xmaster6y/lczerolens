"""
LCZero model builder.
"""

import os
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, List

import onnx
from onnx2pytorch import ConvertModel
from onnx2torch import convert
from torch import nn


class LczeroModel(nn.Module, metaclass=ABCMeta):
    """
    Class to represent an LCZero model.
    """

    @abstractmethod
    def forward(self, *args: List[Any]):
        """
        Runs the model.
        """
        pass


class LczeroOnnx(LczeroModel):
    def __init__(self, onnx_model: onnx.ModelProto):
        """
        Initializes the model.
        """
        super().__init__()
        self.onnx_model = onnx_model
        try:
            self.convert_model = convert(onnx_model)
        except Exception:
            warnings.warn(
                "Could not convert model using onnx2torch, "
                "trying onnx2pytorch."
            )
            try:
                self.convert_model = ConvertModel(onnx_model)
            except Exception:
                raise ValueError("Could not convert model.")
        self.convert_model.eval()

    def forward(self, *args: List[Any]):
        """
        Runs the model.
        """
        return self.convert_model(*args)


class LczeroResNet(LczeroModel):
    pass


class LczeroSmolgen(LczeroModel):
    pass


class AutoBuilder:
    """
    Class for automatically building a model.
    """

    @classmethod
    def build_from_path(cls, model_path: str):
        """
        Builds a model from a given path.
        """
        if model_path.endswith(".onnx"):
            return cls.build_from_onnx(model_path)
        else:
            raise NotImplementedError(
                f"Model path {model_path} is not supported."
            )

    @classmethod
    def build_from_onnx(cls, model_path: str):
        """
        Builds a model from a given path.
        """
        if not os.path.exists(model_path):
            raise FileExistsError(f"Model path {model_path} does not exist.")
        try:
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
        except Exception:
            raise ValueError(f"Could not load model at {model_path}.")
        return LczeroOnnx(onnx_model)
