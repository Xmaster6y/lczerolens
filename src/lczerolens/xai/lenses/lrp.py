"""Compute LRP heatmap for a given model and input."""

from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import chess
import onnx2torch
import torch
from torch.utils.data import DataLoader, Dataset
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.composites import Composite, LayerMapComposite
from zennit.rules import Epsilon, Pass, ZPlus
from zennit.types import Activation

from lczerolens.model.wrapper import ModelWrapper
from lczerolens.xai.helpers import lrp as lrp_helpers
from lczerolens.xai.lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@Lens.register("lrp")
class LrpLens(Lens):
    """Class for wrapping the LCZero models."""

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """Returns whether the lens is compatible with the model.

        Parameters
        ----------
        wrapper : ModelWrapper
            The model wrapper.

        Returns
        -------
        bool
            Whether the lens is compatible with the model.
        """
        return isinstance(wrapper.model, torch.nn.Module)

    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """Runs basic LRP on the model.

        Parameters
        ----------
        board : chess.Board
            The board to compute the heatmap for.
        wrapper : ModelWrapper
            The model wrapper.

        Returns
        -------
        torch.Tensor
            The heatmap for the given board.
        """
        composite = kwargs.get("composite", None)
        target = kwargs.get("target", "policy")
        replace_onnx2torch = kwargs.get("replace_onnx2torch", True)
        linearise_softmax = kwargs.get("linearise_softmax", False)
        init_rel_fn = kwargs.get("init_rel_fn", None)
        relevance = self._compute_lrp_relevance(
            [board],
            wrapper,
            composite=composite,
            target=target,
            replace_onnx2torch=replace_onnx2torch,
            linearise_softmax=linearise_softmax,
            init_rel_fn=init_rel_fn,
        )
        return relevance[0]

    def analyse_dataset(
        self,
        dataset: Dataset,
        wrapper: ModelWrapper,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        save_to: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """Cache the activations for a given model and dataset."""
        if save_to is not None:
            raise NotImplementedError("Saving to file is not implemented.")
        composite = kwargs.get("composite", None)
        target = kwargs.get("target", "policy")
        replace_onnx2torch = kwargs.get("replace_onnx2torch", True)
        linearise_softmax = kwargs.get("linearise_softmax", False)
        init_rel_fn = kwargs.get("init_rel_fn", None)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        relevances = {}
        for batch in dataloader:
            inidices, boards = batch
            batched_relevances = self._compute_lrp_relevance(
                boards,
                wrapper,
                composite=composite,
                target=target,
                replace_onnx2torch=replace_onnx2torch,
                linearise_softmax=linearise_softmax,
                init_rel_fn=init_rel_fn,
            )
            for idx, relevance in zip(inidices, batched_relevances):
                relevances[idx] = relevance
        return relevances

    def _compute_lrp_relevance(
        self,
        boards: List[chess.Board],
        wrapper: ModelWrapper,
        composite: Optional[Any] = None,
        target: Optional[str] = None,
        replace_onnx2torch: bool = True,
        linearise_softmax: bool = False,
        init_rel_fn: Optional[Callable] = None,
    ):
        """
        Compute LRP heatmap for a given model and input.
        """

        with self.context(wrapper, composite, replace_onnx2torch, linearise_softmax) as modified_model:
            output, input_tensor = modified_model.predict(
                boards,
                with_grad=True,
                input_requires_grad=True,
                return_input=True,
            )
            if target is None:
                output.backward(gradient=(output if init_rel_fn is None else init_rel_fn(output)))
            else:
                output[target].backward(
                    gradient=(output[target] if init_rel_fn is None else init_rel_fn(output[target]))
                )
        return input_tensor.grad

    @staticmethod
    def make_default_composite():
        canonizers = [SequentialMergeBatchNorm()]

        layer_map = [
            (Activation, Pass()),
            (torch.nn.Conv2d, ZPlus()),
            (torch.nn.Linear, Epsilon(epsilon=1e-6)),
            (torch.nn.AdaptiveAvgPool2d, Epsilon(epsilon=1e-6)),
        ]
        return LayerMapComposite(layer_map=layer_map, canonizers=canonizers)

    @staticmethod
    @contextmanager
    def context(
        wrapper: ModelWrapper,
        composite: Optional[Composite] = None,
        replace_onnx2torch: bool = True,
        linearise_softmax: bool = False,
    ):
        """Context manager for the lens."""
        if composite is None:
            composite = LrpLens.make_default_composite()

        new_module_mapping = {}
        old_module_mapping = {}

        for name, module in wrapper.model.named_modules():
            if linearise_softmax:
                if isinstance(module, torch.nn.Softmax):
                    new_module_mapping[name] = torch.nn.Identity()
                    old_module_mapping[name] = module
            if replace_onnx2torch:
                if isinstance(module, onnx2torch.node_converters.OnnxBinaryMathOperation):
                    if module.math_op_function is torch.add:
                        new_module_mapping[name] = lrp_helpers.AddEpsilon()
                        old_module_mapping[name] = module
                    elif module.math_op_function is torch.mul:
                        new_module_mapping[name] = lrp_helpers.MulUniform()
                        old_module_mapping[name] = module
                elif isinstance(module, onnx2torch.node_converters.OnnxMatMul):
                    new_module_mapping[name] = lrp_helpers.MatMulEpsilon()
                    old_module_mapping[name] = module
                elif isinstance(module, onnx2torch.node_converters.OnnxFunction):
                    if module.function is torch.tanh:
                        new_module_mapping[name] = torch.nn.Tanh()
                        old_module_mapping[name] = module
                elif isinstance(
                    module,
                    onnx2torch.node_converters.OnnxGlobalAveragePoolWithKnownInputShape,  # noqa
                ):
                    new_module_mapping[name] = torch.nn.AdaptiveAvgPool2d(1)
                    old_module_mapping[name] = module
        for name, module in new_module_mapping.items():
            setattr(wrapper.model, name, module)

        with composite.context(wrapper) as modified_model:
            yield modified_model

        for name, module in old_module_mapping.items():
            setattr(wrapper.model, name, module)
