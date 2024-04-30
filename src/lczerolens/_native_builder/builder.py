"""LCZero model builder."""

import os
import re

import torch
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.utils.safe_shape_inference import safe_shape_inference
from tensordict import TensorDict
from torch import nn

from .senet import SeNet


class BuilderError(Exception):
    """Error raised when the builder fails."""


class NativeBuilder:
    """Class for automatically building a model."""

    _module_exp = re.compile(r"\/?(?P<module_name>[a-z_\-]+)(?P<module_index>[0-9]*)" r"(\/(?P<remaining>.*))?")

    @staticmethod
    def _translate(name):
        """
        Translates a name.
        """
        mapping = {
            "b": "bias",
            "w": "weight",
            "kernel": "weight",
            "dense": "linear",
            "matmul": "weight",
            "add": "bias",
            "inputconv": "ini_conv",
            "encoder": "block",
            "ffn": "mlp",
        }
        if name in mapping:
            return mapping[name]
        else:
            return name

    @classmethod
    def _parse_remaining(cls, remaining):
        """
        Finds the submodule name.
        """
        if remaining is None:
            raise BuilderError("Could not match None")
        match = cls._module_exp.match(remaining.lower())
        if match is None:
            raise BuilderError(f"Could not match {remaining}")
        module_name = cls._translate(match.group("module_name"))
        module_index = match.group("module_index")
        remaining = cls._translate(match.group("remaining"))
        return module_name, module_index, remaining

    @classmethod
    def build_from_path(cls, model_path: str):
        """
        Builds a model from a given path.
        """
        if model_path.endswith(".onnx"):
            return cls.build_from_onnx_path(model_path)
        elif model_path.endswith(".pt"):
            return cls.build_from_torch_path(model_path)
        else:
            raise NotImplementedError(f"Model path {model_path} is not supported.")

    @classmethod
    def build_from_onnx_path(cls, onnx_model_path: str):
        """
        Builds a model from a given path.
        """
        if not os.path.exists(onnx_model_path):
            raise FileExistsError(f"Model path {onnx_model_path} does not exist.")
        try:
            onnx_model = safe_shape_inference(onnx_model_path)
            onnx_graph = OnnxGraph(onnx_model.graph)
        except Exception:
            raise BuilderError(f"Could not load model at {onnx_model_path}.")
        for name, _ in onnx_graph.initializers.items():
            match = cls._module_exp.match(name)
            if match:
                module_name = match.group("module_name")
                if module_name == "block":
                    return cls._build_senet_from_onnx(onnx_graph)
        raise BuilderError(f"Could not load model at {onnx_model_path}.")

    @staticmethod
    def make_onnx_td_forward(onnx_model):
        old_forward = onnx_model.forward
        output_node = list(onnx_model.graph.nodes)[-1]
        output_names = [n.name.replace("output_", "") for n in output_node.all_input_nodes]

        def td_forward(x):
            old_out = old_forward(x)
            return TensorDict(
                {name: old_out[i] for i, name in enumerate(output_names)},
                batch_size=x.shape[0],
            )

        return td_forward

    @classmethod
    def build_from_torch_path(cls, torch_model_path: str):
        """
        Builds a model from a given path.
        """
        if not os.path.exists(torch_model_path):
            raise FileExistsError(f"Model path {torch_model_path} does not exist.")
        try:
            torch_model = torch.load(torch_model_path)
        except Exception:
            raise BuilderError(f"Could not load model at {torch_model_path}.")
        if isinstance(torch_model, nn.Module):
            return torch_model
        else:
            raise BuilderError(f"Could not load model at {torch_model_path}.")

    @classmethod
    def _build_senet_from_onnx(cls, onnx_graph):
        """
        Builds a SeNet from an onnx graph.
        """
        state_dict = {}
        n_blocks = 0
        n_hidden = None
        n_hidden_red = None
        heads = None
        convert_value_to_wdl = False
        for name, onnx_tensor in onnx_graph.initializers.items():
            parsed_name = name.replace("/w", "")
            try:
                module_name, module_index, remaining = cls._parse_remaining(parsed_name)
            except BuilderError:
                continue

            if module_name == "ini_conv":
                state_dict_name = f"ini_conv.{remaining}"
            elif module_name == "block":
                if "axes" in remaining:
                    continue
                if int(module_index) >= n_blocks:
                    n_blocks = int(module_index) + 1
                if remaining.startswith("conv2/se"):
                    remaining = remaining.replace("conv2/se", "")
                    (
                        submodule_name,
                        submodule_index,
                        subremaining,
                    ) = cls._parse_remaining(remaining)
                    state_dict_name = f"block{module_index}.se_layer." f"linear{submodule_index}.{submodule_name}"
                else:
                    (
                        submodule_name,
                        submodule_index,
                        subremaining,
                    ) = cls._parse_remaining(remaining)
                    state_dict_name = f"block{module_index}." f"{submodule_name}{submodule_index}.{subremaining}"
            elif module_name in ["mlh", "wdl", "policy", "value"]:
                if heads is None:
                    heads = [module_name]
                elif module_name not in heads:
                    heads.append(module_name)
                (
                    submodule_name,
                    submodule_index,
                    subremaining,
                ) = cls._parse_remaining(remaining)
                state_dict_name = f"{module_name}." f"{submodule_name}{submodule_index}.{subremaining}"
            elif module_name == "const":
                continue
            else:
                raise BuilderError(f"Could not match {module_name}")

            if "linear" in state_dict_name and "weight" in state_dict_name:
                torch_tensor = onnx_tensor.to_torch().transpose(1, 0)
            else:
                torch_tensor = onnx_tensor.to_torch()

            if "se_layer.linear1.weight" in state_dict_name:
                tmp_n_hidden_red, tmp_n_hidden = torch_tensor.shape
                if n_hidden is None:
                    n_hidden = tmp_n_hidden
                elif n_hidden != tmp_n_hidden:
                    raise BuilderError("n_hidden mismatch: " f"{n_hidden} != {tmp_n_hidden}")
                if n_hidden_red is None:
                    n_hidden_red = tmp_n_hidden_red
                elif n_hidden_red != tmp_n_hidden_red:
                    raise BuilderError("n_hidden_red mismatch: " f"{n_hidden_red} != {tmp_n_hidden_red}")
            if state_dict_name == "value.linear2.bias":
                if torch_tensor.shape[0] != 1:
                    convert_value_to_wdl = True

            state_dict[state_dict_name] = torch_tensor
        if n_hidden is None or n_hidden_red is None or heads is None or n_blocks == 0:
            raise BuilderError("Could not build SeNet from onnx graph.")

        if convert_value_to_wdl:
            if "wdl" in heads:
                raise BuilderError("Inconsistent heads.")
            heads.append("wdl")
            heads.remove("value")
            for key in list(state_dict.keys()):
                if "value" in key:
                    new_key = key.replace("value", "wdl")
                    state_dict[new_key] = state_dict.pop(key)

        model = SeNet(
            n_blocks=n_blocks,
            n_hidden=n_hidden,
            n_hidden_red=n_hidden_red,
            heads=heads,
        )
        model.load_state_dict(state_dict)
        return model
