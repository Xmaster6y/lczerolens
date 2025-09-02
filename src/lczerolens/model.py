"""Class for wrapping the LCZero models."""

import os
from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Union, Iterable, List, Optional, Sequence
import tempfile

import torch
from onnx2torch import convert
from onnx2torch.utils.safe_shape_inference import safe_shape_inference
from tensordict import TensorDict
from torch import nn

from tensordict.nn import TensorDictModule

from lczerolens.board import InputEncoding, LczeroBoard


class LczeroModel(TensorDictModule):
    """Class for wrapping the LCZero models."""

    def __init__(self, module: nn.Module, out_keys: List[str], **kwargs):
        """
        Parameters
        ----------
        module : nn.Module
            The module to wrap.
        out_keys : List[str]
            The keys of the output of the module.
        **kwargs : Any
            Additional keyword arguments to pass to the super().__init__ method.

        Raises
        ------
        ValueError
            If the module is not a valid model type
        """
        if not isinstance(module, nn.Module):
            raise TypeError(f"Got invalid module type {type(module)}. Expected nn.Module.")
        super().__init__(module, ["board"], out_keys, **kwargs)

    def prepare_boards(
        self,
        *boards: LczeroBoard,
        input_encoding: InputEncoding = InputEncoding.INPUT_CLASSICAL_112_PLANE,
    ) -> torch.Tensor:
        """Prepares the boards for the model.

        Parameters
        ----------
        *boards : LczeroBoard
            The boards to prepare.
        input_encoding : InputEncoding, optional
            The encoding of the boards.

        Returns
        -------
        torch.Tensor
            The prepared boards.
        """
        for board in boards:
            if not isinstance(board, LczeroBoard):
                raise ValueError(f"Got invalid input type {type(board)}.")

        tensor_list = [board.to_input_tensor(input_encoding=input_encoding).unsqueeze(0) for board in boards]
        batched_tensor = torch.cat(tensor_list, dim=0)
        batched_tensor = batched_tensor.to(self.device)

        return batched_tensor

    def forward(
        self,
        inputs: Union[TensorDict, LczeroBoard, Iterable[LczeroBoard], torch.Tensor],
        prepare_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> TensorDict:
        """
        Parameters
        ----------
        inputs : Union[TensorDict, Iterable[LczeroBoard], torch.Tensor]
            The inputs to the model.
        prepare_kwargs : Optional[Dict[str, Any]], optional
            Keyword arguments to pass to the prepare_boards method, by default None
        **kwargs : Any
            Additional keyword arguments to pass to the super().forward method.

        Returns
        -------
        TensorDict
            The output of the model.
        """
        prepare_kwargs = prepare_kwargs or {}
        if isinstance(inputs, LczeroBoard):  # TODO: Move to prepare_baords
            inputs = (inputs,)
        if not isinstance(inputs, TensorDict) and not isinstance(inputs, torch.Tensor):
            inputs = self.prepare_boards(*inputs, **prepare_kwargs)
        if not isinstance(inputs, TensorDict):
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(0)
            elif len(inputs.shape) != 4:
                raise ValueError(f"Expected 3D or 4D tensor, got {inputs.shape}.")
            inputs = TensorDict({"board": inputs}, batch_size=inputs.shape[0])
        return super().forward(inputs, **kwargs)

    def _call_module(self, tensors: Sequence[torch.Tensor], **kwargs: Any) -> Sequence[torch.Tensor]:
        out = super()._call_module(tensors, **kwargs)
        return tuple(out)

    @classmethod
    def from_model(cls, model: nn.Module, **kwargs) -> "LczeroModel":
        """Creates a wrapper from a model.

        Parameters
        ----------
        model : nn.Module
            The model to wrap.
        **kwargs : Any
            Additional keyword arguments to pass to the super().__init__ method.

        Returns
        -------
        LczeroModel
            The wrapped model instance
        """
        return cls(model, out_keys=cls._get_output_names(model), **kwargs)

    @classmethod
    def from_path(cls, model_path: str, **kwargs) -> "LczeroModel":
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
            return cls.from_onnx_path(model_path, **kwargs)
        elif model_path.endswith(".pt"):
            return cls.from_torch_path(model_path, **kwargs)
        else:
            raise NotImplementedError(f"Model path {model_path} is not supported.")

    @classmethod
    def from_onnx_path(cls, onnx_model_path: str, check: bool = True, **kwargs) -> "LczeroModel":
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
            return cls.from_model(onnx_torch_model, **kwargs)
        except Exception as e:
            raise ValueError(f"Could not load model at {onnx_model_path}.") from e

    @classmethod
    def from_torch_path(cls, torch_model_path: str, weights_only: bool = False, **kwargs) -> "LczeroModel":
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
            torch_model = torch.load(torch_model_path, weights_only=weights_only)
        except Exception as e:
            raise ValueError(f"Could not load model at {torch_model_path}.") from e
        if isinstance(torch_model, LczeroModel):
            return torch_model
        elif isinstance(torch_model, nn.Module):
            return cls.from_model(torch_model, **kwargs)
        else:
            raise ValueError(f"Could not load model at {torch_model_path}.")

    def push_to_hf(
        self,
        repo_id: str,
        create_if_not_exists: bool = True,
        create_kwargs: Optional[Dict[str, Any]] = None,
        path_in_repo: str = "model.pt",
        **kwargs,
    ):
        """Pushes the model to the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The repository id to push the model to.
        create_if_not_exists : bool, optional
            Whether to create the repository if it does not exist, by default True
        create_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments to pass to the create_repo method.
        path_in_repo : str, optional
            The path in the repository to save the model to.
        **kwargs : Any
            Additional keyword arguments to pass to the upload_file method.

        Raises
        ------
        ImportError
            If the huggingface_hub library is not installed.
        """
        try:
            from huggingface_hub import create_repo, repo_exists, upload_file
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to push the model to the Hugging Face Hub, install it with `pip install lczerolens[hf]`."
            ) from e

        _exists = repo_exists(repo_id, token=create_kwargs.get("token", None))
        if create_if_not_exists and not _exists:
            create_kwargs = create_kwargs or {}
            create_repo(repo_id, **create_kwargs)
        elif not _exists:
            raise ValueError(f"Repository {repo_id} does not exist.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pt")
            torch.save(self.module, path)
            upload_file(path_or_fileobj=path, repo_id=repo_id, path_in_repo=path_in_repo, **kwargs)

    @classmethod
    def from_hf(
        cls, repo_id: str, filename: str = "model.pt", hf_hub_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ) -> "LczeroModel":
        """
        Loads a model from the Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            The repository id to load the model from.
        filename : str
            The filename of the model to load.
        hf_hub_kwargs : Optional[Dict[str, Any]], optional
            Additional keyword arguments to pass to the hf_hub_download method.
        **kwargs : Any
            Additional keyword arguments to pass to the from_path method.

        Returns
        -------
        LczeroModel
            The loaded model instance

        Raises
        ------
        ImportError
            If the huggingface_hub library is not installed.
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required to load the model from the Hugging Face Hub, install it with `pip install lczerolens[hf]`."
            ) from e

        hf_hub_kwargs = hf_hub_kwargs or {}
        path = hf_hub_download(repo_id, filename, **hf_hub_kwargs)
        return cls.from_path(path, **kwargs)

    @staticmethod
    def _get_output_names(model: nn.Module) -> List[str]:
        """Returns the output names of the model.

        Parameters
        ----------
        model : nn.Module
            The model to get the output names from.

        Returns
        -------
        List[str]
            The output names of the model.
        """
        output_node = list(model.graph.nodes)[-1]
        return [n.name.replace("output_", "") for n in output_node.all_input_nodes]


class ForceValue(LczeroModel):
    """Class for forcing and isolating the value flow."""

    def __init__(self, module: nn.Module, out_keys: List[str], **kwargs):
        super().__init__(module, out_keys, **kwargs)
        output_names = self._get_output_names(self.module)
        self._compute_value = "wdl" in output_names
        self._wdl_index = output_names.index("wdl") if self._compute_value else None

    @staticmethod
    def _get_output_names(model: nn.Module) -> List[str]:
        """Returns the output names of the model.

        Parameters
        ----------
        model : nn.Module
            The model to get the output names from.

        Returns
        -------
        List[str]
            The output names of the model.
        """
        names = LczeroModel._get_output_names(model)
        if "value" in names:
            return names
        elif "wdl" in names:
            return names + ["value"]
        else:
            raise ValueError("The model does not have a `value` or `wdl` head.")

    def _call_module(self, tensors: Sequence[torch.Tensor], **kwargs: Any) -> Sequence[torch.Tensor]:
        out = super()._call_module(tensors, **kwargs)
        if self._compute_value:
            wdl = out[self._wdl_index]
            out = (*out, wdl @ torch.tensor([1.0, 0.0, -1.0], device=wdl.device))
        return out


class Flow(LczeroModel, metaclass=ABCMeta):
    """Base class for isolating a flow."""

    def __init__(self, module: nn.Module, out_keys: List[str], **kwargs):
        if self._flow_type not in out_keys:
            raise ValueError(f"The flow type `{self._flow_type}` is not in the output keys ({out_keys=}).")
        filtered_out_keys = [key if key == self._flow_type else "_" for key in out_keys]
        super().__init__(module, filtered_out_keys, **kwargs)

    @property
    @abstractmethod
    def _flow_type(self) -> str:
        """Returns the flow type."""
        pass


class PolicyFlow(Flow):
    """Class for isolating the policy flow."""

    _flow_type = "policy"


class ValueFlow(Flow):
    """Class for isolating the value flow."""

    _flow_type = "value"


class WdlFlow(Flow):
    """Class for isolating the WDL flow."""

    _flow_type = "wdl"


class MlhFlow(Flow):
    """Class for isolating the MLH flow."""

    _flow_type = "mlh"
