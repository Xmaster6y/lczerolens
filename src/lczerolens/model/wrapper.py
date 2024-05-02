"""Class for wrapping the LCZero models."""

import os
from typing import Dict, Iterable, Type, Union

import chess
import torch
from onnx2torch import convert
from onnx2torch.utils.safe_shape_inference import safe_shape_inference
from tensordict import TensorDict
from torch import nn

from lczerolens.encodings import board as board_encodings


class ModelWrapper(nn.Module):
    """Class for wrapping the LCZero models."""

    def __init__(
        self,
        model: nn.Module,
    ):
        """Initializes the wrapper."""
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    @property
    def device(self):
        """Returns the device."""
        return next(self.parameters()).device

    @device.setter
    def device(self, device: torch.device):
        """Sets the device."""
        self.to(device)

    @classmethod
    def from_path(cls, model_path: str):
        """Creates a wrapper from a model path."""
        if model_path.endswith(".onnx"):
            return cls.from_onnx_path(model_path)
        elif model_path.endswith(".pt"):
            return cls.from_torch_path(model_path)
        else:
            raise NotImplementedError(f"Model path {model_path} is not supported.")

    @classmethod
    def from_onnx_path(cls, onnx_model_path: str, check: bool = True):
        """
        Builds a model from a given path.
        """
        if not os.path.exists(onnx_model_path):
            raise FileExistsError(f"Model path {onnx_model_path} does not exist.")
        try:
            if check:
                onnx_model = safe_shape_inference(onnx_model_path)
            onnx_torch_model = convert(onnx_model)
            onnx_torch_model.forward = cls.make_onnx_td_forward(onnx_torch_model)
            return cls(model=onnx_torch_model)
        except Exception:
            raise ValueError(f"Could not load model at {onnx_model_path}.")

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
    def from_torch_path(cls, torch_model_path: str):
        """
        Builds a model from a given path.
        """
        if not os.path.exists(torch_model_path):
            raise FileExistsError(f"Model path {torch_model_path} does not exist.")
        try:
            torch_model = torch.load(torch_model_path)
        except Exception:
            raise ValueError(f"Could not load model at {torch_model_path}.")
        if isinstance(torch_model, ModelWrapper):
            return torch_model
        elif isinstance(torch_model, nn.Module):
            return cls(model=torch_model)
        else:
            raise ValueError(f"Could not load model at {torch_model_path}.")

    def predict(
        self,
        to_pred: Union[chess.Board, Iterable[chess.Board]],
        with_grad: bool = False,
        input_requires_grad: bool = False,
        return_input: bool = False,
    ):
        """Predicts the move."""
        if isinstance(to_pred, chess.Board):
            board_list = [to_pred]
        elif isinstance(to_pred, Iterable):
            board_list = to_pred  # type: ignore
        else:
            raise ValueError("Invalid input type.")

        tensor_list = [board_encodings.board_to_input_tensor(board).unsqueeze(0) for board in board_list]
        batched_tensor = torch.cat(tensor_list, dim=0)
        if input_requires_grad:
            batched_tensor.requires_grad = True
        batched_tensor = batched_tensor.to(self.device)
        with torch.set_grad_enabled(with_grad):
            out = self.forward(batched_tensor)

        if return_input:
            return out, batched_tensor
        return (out,)


class Flow(ModelWrapper):
    """Class for isolating a flow."""

    _flow_type: str
    all_flows: Dict[str, Type["Flow"]] = {}

    def __init__(
        self,
        model: nn.Module,
    ):
        if not self.is_compatible(model):
            raise ValueError(f"The model does not have a {self._flow_type} head.")
        super().__init__(model=model)

    @classmethod
    def register(cls, name: str):
        """Registers the flow."""

        def decorator(subclass):
            cls.all_flows[name] = subclass
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str, **kwargs) -> "Flow":
        """Returns the flow from its name."""
        return cls.all_flows[name](**kwargs)

    @classmethod
    def get_subclass(cls, name: str) -> Type["Flow"]:
        """Returns the subclass."""
        return cls.all_flows[name]

    @classmethod
    def is_compatible(cls, model: nn.Module):
        return hasattr(model, cls._flow_type) or hasattr(model, f"output/{cls._flow_type}")

    def forward(self, x):
        """Forward pass."""
        return self.model(x)[self._flow_type]


@Flow.register("policy")
class PolicyFlow(Flow):
    """Class for isolating the policy flow."""

    _flow_type = "policy"


@Flow.register("value")
class ValueFlow(Flow):
    """Class for isolating the value flow."""

    _flow_type = "value"


@Flow.register("wdl")
class WdlFlow(Flow):
    """Class for isolating the WDL flow."""

    _flow_type = "wdl"


@Flow.register("mlh")
class MlhFlow(Flow):
    """Class for isolating the MLH flow."""

    _flow_type = "mlh"
