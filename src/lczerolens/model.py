"""Class for wrapping the LCZero models."""

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import tempfile

import torch
from onnx2torch import convert
from onnx2torch.utils.safe_shape_inference import safe_shape_inference
from torch import nn

from tensordict.nn import TensorDictModule

from lczerolens.schema import LczeroKeys, _NETWORK_HEAD_KEYS

MISSING_HF_ERROR = (
    "huggingface_hub is required to push or load the model from the Hugging Face Hub. "
    "Install it with `pip install lczerolens[hub]` or directly via `pip install huggingface_hub`."
)


class LczeroModel(TensorDictModule):
    """Class for wrapping the LCZero models."""

    def __init__(
        self,
        module: nn.Module,
        out_keys: List[str],
        *,
        network: str | None = None,
        network_checksum: str | None = None,
        **kwargs,
    ):
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
        heads = tuple(out_keys)
        if not heads or any(head not in _NETWORK_HEAD_KEYS for head in heads):
            raise ValueError("LczeroModel supports only policy, wdl, value, and mlh output heads.")
        if len(heads) != len(set(heads)):
            raise ValueError("LczeroModel output heads must be unique.")
        self.heads = heads
        self.network = network
        self.network_checksum = network_checksum
        super().__init__(
            module,
            [LczeroKeys.INPUT_PLANES],
            [_NETWORK_HEAD_KEYS[head] for head in heads],
            **kwargs,
        )

    def _call_module(self, tensors: Sequence[torch.Tensor], **kwargs: Any) -> Sequence[torch.Tensor]:
        out = super()._call_module(tensors, **kwargs)
        # TensorDictModule expects one value per output key.  Converted lc0
        # graphs return a tuple, while transparent PyTorch wrappers often
        # return a single tensor.
        return (out,) if isinstance(out, torch.Tensor) else tuple(out)

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
        out_keys = kwargs.pop("out_keys", None)
        return cls(model, out_keys=out_keys or cls._get_output_names(model), **kwargs)

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
            onnx_model = safe_shape_inference(onnx_model_path) if check else onnx_model_path
            onnx_torch_model = convert(onnx_model)
            model = cls.from_model(onnx_torch_model, **kwargs)
            return cls._record_source(model, onnx_model_path)
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
            return cls._record_source(torch_model, torch_model_path)
        elif isinstance(torch_model, nn.Module):
            model = cls.from_model(torch_model, **kwargs)
            return cls._record_source(model, torch_model_path)
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
            raise ImportError(MISSING_HF_ERROR) from e

        create_kwargs = create_kwargs or {}
        _exists = repo_exists(repo_id, token=create_kwargs.get("token", None))
        if create_if_not_exists and not _exists:
            create_repo(repo_id, **create_kwargs)
        elif not _exists:
            raise ValueError(f"Repository {repo_id} does not exist.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "model.pt")
            torch.save(self, path)
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
            raise ImportError(MISSING_HF_ERROR) from e

        hf_hub_kwargs = hf_hub_kwargs or {}
        path = hf_hub_download(repo_id, filename, **hf_hub_kwargs)
        model = cls.from_path(path, **kwargs)
        revision = hf_hub_kwargs.get("revision")
        suffix = f"@{revision}" if revision is not None else ""
        model.network = f"hf://{repo_id}/{filename}{suffix}"
        model.network_checksum = cls._checksum(path)
        return model

    @classmethod
    def _record_source(cls, model: "LczeroModel", path: str) -> "LczeroModel":
        if model.network is None:
            model.network = Path(path).name
        if model.network_checksum is None:
            model.network_checksum = cls._checksum(path)
        return model

    @staticmethod
    def _checksum(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

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
        if not hasattr(model, "graph"):
            raise ValueError(
                "Cannot infer evaluator heads from this PyTorch module. "
                "Pass explicit out_keys to LczeroModel(...) or load an lc0 ONNX model."
            )
        output_node = list(model.graph.nodes)[-1]
        return [n.name.replace("output_", "") for n in output_node.all_input_nodes]
