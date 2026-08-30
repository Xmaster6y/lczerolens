"""TensorDict-centered chess evaluator facade."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, overload, runtime_checkable

import chess
import torch
from tensordict import TensorDict, TensorDictBase

from lczerolens._codec import InputFormat, POLICY_SIZE, encode_input, legal_mask
from lczerolens.evaluation import Evaluation, EvaluationBatch
from lczerolens.model import LczeroModel
from lczerolens.provenance import EvaluationProvenance
from lczerolens.schema import LczeroKeys, _NETWORK_HEAD_KEYS


@runtime_checkable
class Evaluator(Protocol):
    """Structural contract for evaluating one chess position.

    Implementations may expose richer batching or instrumentation APIs, but
    consumers of this protocol request one position and receive one validated
    :class:`Evaluation`.
    """

    def evaluate(self, board: chess.Board, /) -> Evaluation:
        """Evaluate one position and return its standardized evidence."""
        ...


class LczeroEvaluator:
    """Prepare chess positions, execute a model, and standardize its heads."""

    def __init__(
        self,
        model: LczeroModel,
        *,
        input_format: InputFormat = InputFormat.CLASSICAL_112,
        provenance: EvaluationProvenance | None = None,
    ):
        if not isinstance(model, LczeroModel):
            raise TypeError("model must be an LczeroModel.")
        if not isinstance(input_format, InputFormat):
            raise TypeError("input_format must be an InputFormat.")
        if provenance is not None and not isinstance(provenance, EvaluationProvenance):
            raise TypeError("provenance must be EvaluationProvenance when provided.")
        heads = model.heads
        if "policy" not in heads or any(head not in _NETWORK_HEAD_KEYS for head in heads):
            raise ValueError("LczeroEvaluator requires a policy head and only supports policy, wdl, value, and mlh.")
        self.model = model
        self.input_format = input_format
        self.provenance = provenance or _model_provenance(model)

    @classmethod
    def from_path(
        cls,
        model_path: str,
        *,
        input_format: InputFormat = InputFormat.CLASSICAL_112,
        provenance: EvaluationProvenance | None = None,
        **model_kwargs,
    ) -> "LczeroEvaluator":
        """Load a model using the current Lczero model-format adapters."""
        model = LczeroModel.from_path(model_path, **model_kwargs)
        resolved_provenance = provenance or _model_provenance(model)
        return cls(model, input_format=input_format, provenance=resolved_provenance)

    @property
    def device(self) -> torch.device:
        """Device of the wrapped neural module."""
        return self.model.device

    def prepare(self, boards: Sequence[chess.Board]) -> TensorDict:
        """Encode a non-empty position batch as the canonical input TensorDict."""
        resolved = _validate_boards(boards)
        planes = torch.stack([encode_input(board, input_format=self.input_format) for board in resolved]).to(
            self.device
        )
        masks = torch.stack([legal_mask(board) for board in resolved]).to(self.device)
        return TensorDict(
            {LczeroKeys.INPUT_PLANES: planes, LczeroKeys.INPUT_LEGAL_MASK: masks},
            batch_size=[len(resolved)],
            device=self.device,
        )

    def finish(self, boards: Sequence[chess.Board], tensors: TensorDictBase) -> EvaluationBatch:
        """Validate and standardize network heads without discarding added keys."""
        resolved = _validate_boards(boards)
        if not isinstance(tensors, TensorDictBase):
            raise TypeError("tensors must be a TensorDictBase.")
        if tensors.batch_dims != 1 or tensors.batch_size[0] != len(resolved):
            raise ValueError("TensorDict must have one batch dimension matching the board count.")
        _require_shape(
            tensors,
            LczeroKeys.INPUT_PLANES,
            (len(resolved), 112, 8, 8),
            floating=True,
            finite=True,
        )
        _require_shape(
            tensors,
            LczeroKeys.INPUT_LEGAL_MASK,
            (len(resolved), POLICY_SIZE),
            dtype=torch.bool,
        )
        _require_shape(
            tensors,
            LczeroKeys.NETWORK_POLICY_LOGITS,
            (len(resolved), POLICY_SIZE),
            floating=True,
            finite=True,
        )

        expected_masks = torch.stack([legal_mask(board) for board in resolved]).to(tensors.device)
        if not torch.equal(tensors[LczeroKeys.INPUT_LEGAL_MASK], expected_masks):
            raise ValueError("input/legal_mask does not match the supplied positions.")
        expected_planes = torch.stack([encode_input(board, input_format=self.input_format) for board in resolved]).to(
            tensors.device
        )
        if not torch.equal(tensors[LczeroKeys.INPUT_PLANES], expected_planes):
            raise ValueError("input/planes does not match the supplied positions.")

        logits = tensors[LczeroKeys.NETWORK_POLICY_LOGITS]
        masks = tensors[LczeroKeys.INPUT_LEGAL_MASK]
        probabilities = torch.zeros_like(logits)
        for row in range(len(resolved)):
            if masks[row].any():
                probabilities[row, masks[row]] = torch.softmax(logits[row, masks[row]], dim=0)
        tensors[LczeroKeys.EVALUATION_POLICY] = probabilities

        leaves = set(tensors.keys(include_nested=True, leaves_only=True))
        if LczeroKeys.NETWORK_WDL in leaves:
            _require_shape(
                tensors,
                LczeroKeys.NETWORK_WDL,
                (len(resolved), 3),
                floating=True,
                finite=True,
            )
            wdl = tensors[LczeroKeys.NETWORK_WDL]
            if (
                (wdl < 0).any()
                or (wdl > 1).any()
                or not torch.allclose(
                    wdl.sum(-1), torch.ones(len(resolved), device=wdl.device, dtype=wdl.dtype), atol=1e-5
                )
            ):
                raise ValueError("network/wdl must contain probabilities in [0, 1] that sum to one.")

        if LczeroKeys.NETWORK_VALUE in leaves:
            value = _column(tensors[LczeroKeys.NETWORK_VALUE], len(resolved), "network/value")
            if not torch.isfinite(value).all() or (value.abs() > 1).any():
                raise ValueError("network/value must be finite and in [-1, 1].")
            tensors[LczeroKeys.NETWORK_VALUE] = value
            tensors[LczeroKeys.EVALUATION_VALUE] = value.clone()
        elif LczeroKeys.NETWORK_WDL in leaves:
            wdl = tensors[LczeroKeys.NETWORK_WDL]
            tensors[LczeroKeys.EVALUATION_VALUE] = (wdl[:, 0] - wdl[:, 2]).unsqueeze(-1)

        if LczeroKeys.NETWORK_MLH in leaves:
            mlh = _column(tensors[LczeroKeys.NETWORK_MLH], len(resolved), "network/mlh")
            if not torch.isfinite(mlh).all():
                raise ValueError("network/mlh must be finite.")
            tensors[LczeroKeys.NETWORK_MLH] = mlh
        return EvaluationBatch(resolved, tensors, self.provenance, self.input_format.value)

    @overload
    def evaluate(self, boards: chess.Board, /) -> Evaluation: ...

    @overload
    def evaluate(self, boards: Iterable[chess.Board], /) -> EvaluationBatch: ...

    def evaluate(self, boards: chess.Board | Iterable[chess.Board], /) -> Evaluation | EvaluationBatch:
        """Evaluate one position or a batch through the canonical TensorDict path."""
        single = isinstance(boards, chess.Board)
        resolved = (boards,) if single else tuple(boards)
        tensors = self.prepare(resolved)
        tensors = self.model(tensors)
        batch = self.finish(resolved, tensors)
        return batch[0] if single else batch


def _validate_boards(boards: Sequence[chess.Board]) -> tuple[chess.Board, ...]:
    if isinstance(boards, chess.Board) or not isinstance(boards, Sequence):
        raise TypeError("boards must be a sequence of chess.Board objects.")
    resolved = tuple(boards)
    if not resolved:
        raise ValueError("Expected at least one chess.Board.")
    if any(not isinstance(board, chess.Board) for board in resolved):
        raise TypeError("Every position must be a chess.Board.")
    return resolved


def _require_shape(
    tensors: TensorDictBase,
    key: tuple[str, str],
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype | None = None,
    floating: bool = False,
    finite: bool = False,
) -> None:
    leaves = set(tensors.keys(include_nested=True, leaves_only=True))
    if key not in leaves:
        raise ValueError(f"Missing required TensorDict key {key!r}.")
    value = tensors[key]
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"TensorDict key {key!r} must have shape {shape}.")
    if dtype is not None and value.dtype is not dtype:
        raise ValueError(f"TensorDict key {key!r} must have dtype {dtype}.")
    if floating and not value.is_floating_point():
        raise ValueError(f"TensorDict key {key!r} must have a floating-point dtype.")
    if finite and not torch.isfinite(value).all():
        raise ValueError(f"TensorDict key {key!r} must be finite.")


def _column(value: torch.Tensor, batch_size: int, label: str) -> torch.Tensor:
    if tuple(value.shape) == (batch_size,):
        return value.unsqueeze(-1)
    if tuple(value.shape) == (batch_size, 1):
        return value
    raise ValueError(f"{label} must have shape [{batch_size}] or [{batch_size}, 1].")


def _model_provenance(model: LczeroModel) -> EvaluationProvenance:
    module_type = type(model.module)
    network = model.network
    checksum = model.network_checksum
    return EvaluationProvenance(
        source="lczerolens.LczeroEvaluator",
        model_type=f"{module_type.__module__}.{module_type.__qualname__}",
        network=network,
        network_checksum=checksum,
    )


__all__ = ["Evaluator", "InputFormat", "LczeroEvaluator"]
