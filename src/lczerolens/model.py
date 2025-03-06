"""Class for wrapping the LCZero models."""

import os
from typing import Dict, Type, Any, Tuple, Union, Callable

import torch
from onnx2torch import convert
from onnx2torch.utils.safe_shape_inference import safe_shape_inference
from tensordict import TensorDict
from torch import nn
from nnsight import NNsight
from contextlib import contextmanager

from lczerolens.board import InputEncoding, LczeroBoard


class LczeroModel(NNsight):
    """Class for wrapping the LCZero models."""

    def trace(
        self,
        *inputs: Any,
        **kwargs: Dict[str, Any],
    ):
        kwargs["scan"] = False
        kwargs["validate"] = False
        return super().trace(*inputs, **kwargs)

    def _execute(self, *prepared_inputs: torch.Tensor, **kwargs) -> Any:
        kwargs.pop("input_encoding", None)
        kwargs.pop("input_requires_grad", None)
        with self._ensure_proper_forward():
            return super()._execute(*prepared_inputs, **kwargs)

    def _prepare_inputs(self, *inputs: Union[LczeroBoard, torch.Tensor], **kwargs) -> Tuple[Tuple[Any], int]:
        input_encoding = kwargs.pop("input_encoding", InputEncoding.INPUT_CLASSICAL_112_PLANE)
        input_requires_grad = kwargs.pop("input_requires_grad", False)

        if len(inputs) == 1 and isinstance(inputs[0], torch.Tensor):
            return inputs, len(inputs[0])
        for board in inputs:
            if not isinstance(board, LczeroBoard):
                raise ValueError(f"Got invalid input type {type(board)}.")

        tensor_list = [board.to_input_tensor(input_encoding=input_encoding).unsqueeze(0) for board in inputs]
        batched_tensor = torch.cat(tensor_list, dim=0)
        if input_requires_grad:
            batched_tensor.requires_grad = True
        batched_tensor = batched_tensor.to(self.device)

        return (batched_tensor,), len(inputs)

    def __call__(self, *inputs, **kwargs):
        prepared_inputs, _ = self._prepare_inputs(*inputs, **kwargs)
        return self._execute(*prepared_inputs, **kwargs)

    def __getattr__(self, key):
        if self._envoy._tracer is None:
            return getattr(self._model, key)
        return super().__getattr__(key)

    def __setattr__(self, key, value):
        if (
            (key not in ("_model", "_model_key"))
            and (isinstance(value, torch.nn.Module))
            and (self._envoy._tracer is None)
        ):
            setattr(self._model, key, value)
        else:
            super().__setattr__(key, value)

    @property
    def device(self):
        """Returns the device."""
        return next(self.parameters()).device

    @device.setter
    def device(self, device: torch.device):
        """Sets the device."""
        self.to(device)

    @classmethod
    def from_path(cls, model_path: str) -> "LczeroModel":
        """Creates a wrapper from a model path.

        Parameters
        ----------
        model_path : str
            Path to the model file (.onnx or .pt)

        Returns
        -------
        LczeroModel
            The wrapped model instance

        Raises
        ------
        NotImplementedError
            If the model file extension is not supported
        """
        if model_path.endswith(".onnx"):
            return cls.from_onnx_path(model_path)
        elif model_path.endswith(".pt"):
            return cls.from_torch_path(model_path)
        else:
            raise NotImplementedError(f"Model path {model_path} is not supported.")

    @classmethod
    def from_onnx_path(cls, onnx_model_path: str, check: bool = True) -> "LczeroModel":
        """Builds a model from an ONNX file path.

        Parameters
        ----------
        onnx_model_path : str
            Path to the ONNX model file
        check : bool, optional
            Whether to perform shape inference check, by default True

        Returns
        -------
        LczeroModel
            The wrapped model instance

        Raises
        ------
        FileNotFoundError
            If the model file does not exist
        ValueError
            If the model could not be loaded
        """
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Model path {onnx_model_path} does not exist.")
        try:
            if check:
                onnx_model = safe_shape_inference(onnx_model_path)
            onnx_torch_model = convert(onnx_model)
            return cls(onnx_torch_model)
        except Exception:
            raise ValueError(f"Could not load model at {onnx_model_path}.")

    @classmethod
    def from_torch_path(cls, torch_model_path: str) -> "LczeroModel":
        """Builds a model from a PyTorch file path.

        Parameters
        ----------
        torch_model_path : str
            Path to the PyTorch model file

        Returns
        -------
        LczeroModel
            The wrapped model instance

        Raises
        ------
        FileNotFoundError
            If the model file does not exist
        ValueError
            If the model could not be loaded or is not a valid model type
        """
        if not os.path.exists(torch_model_path):
            raise FileNotFoundError(f"Model path {torch_model_path} does not exist.")
        try:
            torch_model = torch.load(torch_model_path)
        except Exception:
            raise ValueError(f"Could not load model at {torch_model_path}.")
        if isinstance(torch_model, LczeroModel):
            return torch_model
        elif isinstance(torch_model, nn.Module):
            return cls(torch_model)
        else:
            raise ValueError(f"Could not load model at {torch_model_path}.")

    @staticmethod
    def _make_onnx_td_forward(onnx_model: nn.Module) -> Callable:
        """Creates a forward function that returns a TensorDict for ONNX models.

        Parameters
        ----------
        onnx_model : nn.Module
            The ONNX model

        Returns
        -------
        Callable
            The forward function that returns a TensorDict
        """
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

    @contextmanager
    def _ensure_proper_forward(self):
        old_forward = self._model.forward

        output_node = list(self._model.graph.nodes)[-1]
        output_names = [n.name.replace("output_", "") for n in output_node.all_input_nodes]

        def td_forward(x):
            old_out = old_forward(x)
            return TensorDict(
                {name: old_out[i] for i, name in enumerate(output_names)},
                batch_size=x.shape[0],
            )

        self._model.forward = td_forward
        yield
        self._model.forward = old_forward


class Flow(LczeroModel):
    """Base class for isolating a flow."""

    _flow_type: str
    _registry: Dict[str, Type["Flow"]] = {}

    def __init__(
        self,
        model_key,
        *args,
        **kwargs,
    ):
        if isinstance(model_key, LczeroModel):
            raise ValueError("Use the `from_model` classmethod to create a flow.")
        if not self.is_compatible(model_key):
            raise ValueError(f"The model does not have a {self._flow_type} head.")
        super().__init__(model_key, *args, **kwargs)

    @classmethod
    def register(cls, name: str):
        """Registers the flow.

        Parameters
        ----------
        name : str
            The name of the flow to register.

        Returns
        -------
        Callable
            Decorator function that registers the flow subclass.

        Raises
        ------
        ValueError
            If the flow name is already registered.
        """

        if name in cls._registry:
            raise ValueError(f"Flow {name} already registered.")

        def decorator(subclass):
            cls._registry[name] = subclass
            subclass._flow_type = name
            return subclass

        return decorator

    @classmethod
    def from_name(cls, name: str, *args, **kwargs) -> "Flow":
        """Returns the flow from its name.

        Parameters
        ----------
        name : str
            The name of the flow to instantiate.
        *args
            Positional arguments passed to flow constructor.
        **kwargs
            Keyword arguments passed to flow constructor.

        Returns
        -------
        Flow
            The instantiated flow.

        Raises
        ------
        KeyError
            If the flow name is not found.
        """
        if name not in cls._registry:
            raise KeyError(f"Flow {name} not found.")
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def from_model(cls, name: str, model: LczeroModel, *args, **kwargs) -> "Flow":
        """Returns the flow from a model.

        Parameters
        ----------
        name : str
            The name of the flow to instantiate.
        model : LczeroModel
            The model to create the flow from.
        *args
            Positional arguments passed to flow constructor.
        **kwargs
            Keyword arguments passed to flow constructor.

        Returns
        -------
        Flow
            The instantiated flow.

        Raises
        ------
        KeyError
            If the flow name is not found.
        """
        if name not in cls._registry:
            raise KeyError(f"Flow {name} not found.")
        flow_class = cls._registry[name]
        return flow_class(model._model, *args, **kwargs)

    @classmethod
    def is_compatible(cls, model: nn.Module) -> bool:
        """Checks if the model is compatible with this flow.

        Parameters
        ----------
        model : nn.Module
            The model to check compatibility with.

        Returns
        -------
        bool
            Whether the model is compatible with this flow.
        """
        return hasattr(model, cls._flow_type) or hasattr(model, f"output/{cls._flow_type}")

    @contextmanager
    def _ensure_proper_forward(self):
        """Rewrites the forward function to return the flow output."""
        flow_type = getattr(self, "_flow_type", None)
        if flow_type is None:
            return

        with super()._ensure_proper_forward():
            old_forward = self._model.forward

            def flow_forward(*inputs, **kwargs):
                out = old_forward(*inputs, **kwargs)
                return out[flow_type]

            self._model.forward = flow_forward
            yield
            self._model.forward = old_forward


@Flow.register("policy")
class PolicyFlow(Flow):
    """Class for isolating the policy flow."""


@Flow.register("value")
class ValueFlow(Flow):
    """Class for isolating the value flow."""


@Flow.register("wdl")
class WdlFlow(Flow):
    """Class for isolating the WDL flow."""


@Flow.register("mlh")
class MlhFlow(Flow):
    """Class for isolating the MLH flow."""


@Flow.register("force_value")
class ForceValueFlow(Flow):
    """Class for forcing and isolating the value flow."""

    @classmethod
    def is_compatible(cls, model: nn.Module):
        return ValueFlow.is_compatible(model) or WdlFlow.is_compatible(model)

    @contextmanager
    def _ensure_proper_forward(self):
        flow_type = getattr(self, "_flow_type", None)
        if flow_type is None:
            return

        with LczeroModel._ensure_proper_forward(self):
            old_forward = self._model.forward

            def flow_forward(*inputs, **kwargs):
                out = old_forward(*inputs, **kwargs)
                if "value" in out.keys():
                    return out["value"]
                return out["wdl"] @ torch.tensor([1.0, 0.0, -1.0], device=out.device)

            self._model.forward = flow_forward
            yield
            self._model.forward = old_forward
