"""
LCZero model builder.
"""

import os
import re

import torch
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.utils.safe_shape_inference import safe_shape_inference
from torch import nn

from .senet import SeNet
from .vitnet import VitConfig, VitNet


class BuilderError(Exception):
    """
    Error raised when the builder fails.
    """


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
        elif model_path.endswith(".pt"):
            return cls.build_from_torch_path(model_path)
        else:
            raise NotImplementedError(
                f"Model path {model_path} is not supported."
            )

    @classmethod
    def build_from_onnx(cls, onnx_model_or_path: str):
        """
        Builds a model from a given path.
        """
        if not os.path.exists(onnx_model_or_path):
            raise FileExistsError(
                f"Model path {onnx_model_or_path} does not exist."
            )
        try:
            onnx_model = safe_shape_inference(onnx_model_or_path)
            onnx_graph = OnnxGraph(onnx_model.graph)
        except Exception:
            raise BuilderError(
                f"Could not load model at {onnx_model_or_path}."
            )
        block_exp = re.compile(
            r"\/(?P<block_name>[a-z0-9]+)"
            r"\/(?P<module_type>[a-z0-9]+)\/(?P<remaining>.*)"
        )
        for name, _ in onnx_graph.initializers.items():
            match = block_exp.match(name)
            if match:
                module_type = match.group("module_type")
                if module_type in ["mha", "smolgen"]:
                    return cls._build_vitnet_from_onnx(onnx_graph)
                elif module_type in ["conv1", "conv2"]:
                    return cls._build_senet_from_onnx(onnx_graph)
        raise BuilderError(f"Could not load model at {onnx_model_or_path}.")

    @classmethod
    def build_from_torch_path(cls, torch_model_path: str):
        """
        Builds a model from a given path.
        """
        if not os.path.exists(torch_model_path):
            raise FileExistsError(
                f"Model path {torch_model_path} does not exist."
            )
        try:
            torch_model = torch.load(torch_model_path)
        except Exception:
            raise BuilderError(f"Could not load model at {torch_model_path}.")
        if isinstance(torch_model, nn.Module):
            return torch_model
        elif isinstance(torch_model, dict):
            return cls.build_from_state_dict(torch_model)
        else:
            raise BuilderError(f"Could not load model at {torch_model_path}.")

    @classmethod
    def build_from_state_dict(cls, state_dict: dict):
        """
        Builds a model from a given state dict.
        """
        if not isinstance(state_dict, dict):
            raise BuilderError(
                f"State dict must be a dict, not {type(state_dict)}"
            )
        pass

    @staticmethod
    def _build_vitnet_from_onnx(onnx_graph):
        """
        Builds a VitNet from an onnx graph.
        """
        config = VitConfig()
        model = VitNet(config)
        return model

    @staticmethod
    def _build_senet_from_onnx(onnx_graph):
        """
        Builds a SeNet from an onnx graph.
        """
        state_dict = {}
        n_blocks = 0
        n_hidden = None
        n_hidden_red = None
        heads = None
        block_exp = re.compile(
            r"\/(?P<block_name>[a-z]+)(?P<block_index>[0-9]*)"
            r"\/(?P<module_name>[a-z]+)(?P<module_index>[0-9]*)"
            r"(\/(?P<remaining>.*))?"
        )
        se_exp = re.compile(
            r"se\/(?P<module_name>[a-z]+)(?P<module_index>[0-9]*)"
        )
        translate = {
            "bias": "bias",
            "weight": "weight",
            "conv": "conv",
            "b": "bias",
            "w": "weight",
            "kernel": "weight",
            "dense": "linear",
            "matmul": "weight",
            "add": "bias",
        }
        for name, onnx_tensor in onnx_graph.initializers.items():
            parsed_name = name.replace("/w", "")
            match = block_exp.match(parsed_name)
            if match:
                block_name = match.group("block_name")
                if block_name == "const":
                    continue
                block_index = match.group("block_index")
                remaining = match.group("remaining")
                if remaining is not None and remaining.endswith("axes"):
                    continue
                module_name = translate[match.group("module_name")]
                module_index = match.group("module_index")
                if block_name == "inputconv":
                    state_dict[
                        f"ini_conv.{module_name}"
                    ] = onnx_tensor.to_torch()
                    continue
                elif block_name == "block":
                    if block_index == "":
                        raise BuilderError("Block index is empty.")
                    if int(block_index) >= n_blocks:
                        n_blocks = int(block_index) + 1
                    if remaining.startswith("se"):
                        se_match = se_exp.match(remaining)
                        if match is None:
                            raise BuilderError(f"Could not match {remaining}")
                        param_name = translate[se_match.group("module_name")]
                        module_index = se_match.group("module_index")
                        if param_name == "weight" and module_index == "1":
                            (
                                tmp_n_hidden,
                                tmp_n_hidden_red,
                            ) = onnx_tensor.to_torch().shape
                            if n_hidden is None:
                                n_hidden = tmp_n_hidden
                            elif n_hidden != tmp_n_hidden:
                                raise BuilderError(
                                    "n_hidden mismatch: "
                                    f"{n_hidden} != {tmp_n_hidden}"
                                )
                            if n_hidden_red is None:
                                n_hidden_red = tmp_n_hidden_red
                            elif n_hidden_red != tmp_n_hidden_red:
                                raise BuilderError(
                                    "n_hidden_red mismatch: "
                                    f"{n_hidden_red} != {tmp_n_hidden_red}"
                                )
                        module_name = "se_layer.linear"
                        remaining = param_name
                elif block_name == "policy":
                    if heads is None:
                        heads = ["policy"]
                    elif "policy" not in heads:
                        heads.append("policy")
                elif block_name == "value":
                    if heads is None:
                        heads = ["value"]
                    elif "value" not in heads:
                        heads.append("value")
                elif block_name == "const":
                    continue
                else:
                    raise BuilderError(f"Could not match {block_name}")
                param_name = translate[remaining]
                if "linear" in module_name and param_name == "weight":
                    torch_tensor = onnx_tensor.to_torch().transpose(1, 0)
                else:
                    torch_tensor = onnx_tensor.to_torch()
                state_dict[
                    f"{block_name}{block_index}."
                    f"{module_name}{module_index}.{param_name}"
                ] = torch_tensor
        if (
            n_hidden is None
            or n_hidden_red is None
            or heads is None
            or n_blocks == 0
        ):
            raise BuilderError("Could not build SeNet from onnx graph.")
        model = SeNet(
            n_blocks=n_blocks,
            n_hidden=n_hidden,
            n_hidden_red=n_hidden_red,
            heads=heads,
        )
        model.load_state_dict(state_dict)
        return model
