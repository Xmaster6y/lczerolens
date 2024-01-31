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

    _module_exp = re.compile(
        r"\/?(?P<module_name>[a-z_\-]+)(?P<module_index>[0-9]*)"
        r"(\/(?P<remaining>.*))?"
    )

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
        for name, _ in onnx_graph.initializers.items():
            match = cls._module_exp.match(name)
            if match:
                module_name = match.group("module_name")
                if module_name == "encoder":
                    return cls._build_vitnet_from_onnx(onnx_graph)
                elif module_name == "block":
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
                module_name, module_index, remaining = cls._parse_remaining(
                    parsed_name
                )
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
                    state_dict_name = (
                        f"block{module_index}.se_layer."
                        f"linear{submodule_index}.{submodule_name}"
                    )
                else:
                    (
                        submodule_name,
                        submodule_index,
                        subremaining,
                    ) = cls._parse_remaining(remaining)
                    state_dict_name = (
                        f"block{module_index}."
                        f"{submodule_name}{submodule_index}.{subremaining}"
                    )
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
                state_dict_name = (
                    f"{module_name}."
                    f"{submodule_name}{submodule_index}.{subremaining}"
                )
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
                    raise BuilderError(
                        "n_hidden mismatch: " f"{n_hidden} != {tmp_n_hidden}"
                    )
                if n_hidden_red is None:
                    n_hidden_red = tmp_n_hidden_red
                elif n_hidden_red != tmp_n_hidden_red:
                    raise BuilderError(
                        "n_hidden_red mismatch: "
                        f"{n_hidden_red} != {tmp_n_hidden_red}"
                    )
            if state_dict_name == "value.linear2.bias":
                if torch_tensor.shape[0] != 1:
                    convert_value_to_wdl = True

            state_dict[state_dict_name] = torch_tensor
        if (
            n_hidden is None
            or n_hidden_red is None
            or heads is None
            or n_blocks == 0
        ):
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

    @classmethod
    def _build_vitnet_from_onnx(cls, onnx_graph):
        """
        Builds a VitNet from an onnx graph.
        """
        state_dict = {}
        n_blocks = 0
        n_heads = None
        n_hidden = None
        n_hidden_red_block = None
        n_hidden_red_mlh = None
        n_hidden_red_wdl = None
        mlp_scale = None
        attention_scale = None
        policy_scale = None
        act_mode = None
        heads = None
        convert_value_to_wdl = False
        smolgen_decompress = None
        positional_encodings = None
        for name, onnx_tensor in onnx_graph.initializers.items():
            parsed_name = name.replace("/w/w", "/matmul")
            parsed_name = parsed_name.replace("/w", "")
            is_mha_shape = False
            is_smolgen_decompress = False
            is_positional_encodings = False
            module_name, module_index, remaining = cls._parse_remaining(
                parsed_name
            )
            if module_name == "attn_body":
                submodule_name, _, _ = cls._parse_remaining(remaining)
                if submodule_name == "batch":
                    continue
                state_dict_name = f"ini_linear.{submodule_name}"
            elif module_name == "ip_mul_gate":
                state_dict_name = "ini_multiply.weight"
            elif module_name == "ip_add_gate":
                state_dict_name = "ini_multiply.bias"
            elif module_name == "block":
                if int(module_index) >= n_blocks:
                    n_blocks = int(module_index) + 1
                (
                    submodule_name,
                    submodule_index,
                    subremaining,
                ) = cls._parse_remaining(remaining)
                if submodule_name == "mha":
                    (
                        subsubmodule_name,
                        _,
                        subsubremaining,
                    ) = cls._parse_remaining(
                        subremaining.replace("/dense", "")
                    )
                    submodule_name = f"{subsubmodule_name}_proj"
                    subremaining = subsubremaining
                elif submodule_name == "smolgen":
                    (
                        subsubmodule_name,
                        subsubmodule_index,
                        subsubremaining,
                    ) = cls._parse_remaining(subremaining)
                    if subsubmodule_name == "compress":
                        subsubremaining = "weight"
                    submodule_name = (
                        f"smolgen.{subsubmodule_name}{subsubmodule_index}"
                    )
                    subremaining = subsubremaining
                elif submodule_name == "mlp":
                    (
                        subsubmodule_name,
                        subsubmodule_index,
                        subsubremaining,
                    ) = cls._parse_remaining(subremaining)
                    submodule_name = (
                        f"mlp.{subsubmodule_name}{subsubmodule_index}"
                    )
                    subremaining = subsubremaining

                elif submodule_name == "alpha":
                    tmp_mlp_scale = onnx_tensor.to_torch().item()
                    if mlp_scale is None:
                        mlp_scale = tmp_mlp_scale
                    elif mlp_scale != tmp_mlp_scale:
                        raise BuilderError(
                            "mlp_scale mismatch: "
                            f"{mlp_scale} != {tmp_mlp_scale}"
                        )
                    continue
                elif submodule_name == "ln":
                    (
                        subsubmodule_name,
                        subsubmodule_index,
                        subsubremaining,
                    ) = cls._parse_remaining(subremaining)
                else:
                    raise BuilderError(f"Could not match {submodule_name}")
                state_dict_name = (
                    f"block{module_index}."
                    f"{submodule_name}{submodule_index}.{subremaining}"
                )
            elif module_name in ["const"]:
                if "mha/shape" in remaining:
                    is_mha_shape = True
                elif "smolgen_w" in remaining:
                    is_smolgen_decompress = True
                elif remaining == "pos_encoding":
                    is_positional_encodings = True
                else:
                    continue
                state_dict_name = ""
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
                if module_name == "policy":
                    if submodule_name == "scale":
                        tmp_policy_scale = onnx_tensor.to_torch().item()
                        if policy_scale is None:
                            policy_scale = tmp_policy_scale
                        elif policy_scale != tmp_policy_scale:
                            raise BuilderError(
                                "policy_scale mismatch: "
                                f"{policy_scale} != {tmp_policy_scale}"
                            )
                        continue
                    elif submodule_name == "promotion":
                        if subremaining == "weight":
                            submodule_name = "linear4"
                        else:
                            continue
                    elif submodule_name == "q":
                        submodule_name = "linear2"
                    elif submodule_name == "k":
                        submodule_name = "linear3"
                state_dict_name = (
                    f"{module_name}."
                    f"{submodule_name}{submodule_index}.{subremaining}"
                )
            else:
                raise BuilderError(f"Could not match {module_name}")

            if "weight" in state_dict_name and (
                "linear" in state_dict_name
                or "embed" in state_dict_name
                or "compress" in state_dict_name
            ):
                torch_tensor = onnx_tensor.to_torch().transpose(1, 0)
            else:
                torch_tensor = onnx_tensor.to_torch()

            if "ln" in state_dict_name:
                state_dict_name = state_dict_name.replace("scale", "weight")

            if "qk_proj.scale" in state_dict_name:
                tmp_attention_scale = torch_tensor.item()
                if attention_scale is None:
                    attention_scale = tmp_attention_scale
                elif attention_scale != tmp_attention_scale:
                    raise BuilderError(
                        "attention_scale mismatch: "
                        f"{attention_scale} != {tmp_attention_scale}"
                    )
                continue

            if "smolgen.ln1" in state_dict_name:
                tmp_n_hidden_red_block = torch_tensor.shape[0]
                if n_hidden_red_block is None:
                    n_hidden_red_block = tmp_n_hidden_red_block
                elif n_hidden_red_block != tmp_n_hidden_red_block:
                    raise BuilderError(
                        "n_hidden_red_block mismatch: "
                        f"{n_hidden_red_block} != {tmp_n_hidden_red_block}"
                    )
            elif module_name == "block" and submodule_name == "ln":
                tmp_n_hidden = torch_tensor.shape[0]
                if n_hidden is None:
                    n_hidden = tmp_n_hidden
                elif n_hidden != tmp_n_hidden:
                    raise BuilderError(
                        "n_hidden mismatch: " f"{n_hidden} != {tmp_n_hidden}"
                    )

            if is_mha_shape:
                tmp_n_heads = torch_tensor[2].item()
                if n_heads is None:
                    n_heads = tmp_n_heads
                elif n_heads != tmp_n_heads:
                    raise BuilderError(
                        "n_heads mismatch: " f"{n_heads} != {tmp_n_heads}"
                    )
                continue

            if is_smolgen_decompress:
                smolgen_decompress = torch_tensor
                continue

            if is_positional_encodings:
                positional_encodings = torch_tensor.transpose(1, 0)
                continue

            if state_dict_name == "value.linear2.bias":
                if torch_tensor.shape[0] != 1:
                    convert_value_to_wdl = True

            state_dict[state_dict_name] = torch_tensor

        if convert_value_to_wdl:
            if "wdl" in heads:
                raise BuilderError("Inconsistent heads.")
            heads.append("wdl")
            heads.remove("value")
            for key in list(state_dict.keys()):
                if "value" in key:
                    new_key = key.replace("value", "wdl")
                    state_dict[new_key] = state_dict.pop(key)

        if "wdl" in heads:
            n_hidden_red_wdl = state_dict["wdl.embed.bias"].shape[0]
        if "mlh" in heads:
            n_hidden_red_mlh = state_dict["mlh.embed.bias"].shape[0]

        if "attn_body/mish/softplus" in onnx_graph.nodes:
            act_mode = "softplus-tanh-mul"
        elif "attn_body/relu" in onnx_graph.nodes:
            act_mode = "relu"
        else:
            raise BuilderError("Could not determine activation mode.")

        if (
            n_hidden is None
            or n_heads is None
            or n_hidden_red_block is None
            or n_hidden_red_mlh is None
            or n_hidden_red_wdl is None
            or heads is None
            or n_blocks == 0
            or mlp_scale is None
            or attention_scale is None
            or policy_scale is None
            or act_mode is None
            or smolgen_decompress is None
            or positional_encodings is None
        ):
            raise BuilderError("Could not build VitNet from onnx graph.")
        config = VitConfig(
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_hidden=n_hidden,
            n_hidden_red_block=n_hidden_red_block,
            smolgen_decompress=smolgen_decompress,
            positional_encodings=positional_encodings,
            mlp_scale=mlp_scale,
            attention_scale=attention_scale,
            policy_scale=policy_scale,
            n_hidden_red_mlh=n_hidden_red_mlh,
            n_hidden_red_wdl=n_hidden_red_wdl,
            act_mode=act_mode,
            heads=heads,
        )
        model = VitNet(config)
        model.load_state_dict(state_dict)
        return model
