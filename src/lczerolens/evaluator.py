"""TensorDict-centered chess evaluator facade."""

from __future__ import annotations

from collections.abc import Sequence

import chess
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule

from lczerolens._codec import InputFormat, POLICY_SIZE, encode_input, legal_mask
from lczerolens.evaluation import (
    EVALUATION_POLICY,
    EVALUATION_VALUE,
    INPUT_LEGAL_MASK,
    INPUT_PLANES,
    NETWORK_MLH,
    NETWORK_POLICY_LOGITS,
    NETWORK_VALUE,
    NETWORK_WDL,
    Evaluation,
    EvaluationBatch,
)
from lczerolens.model import LczeroModel


_NETWORK_KEYS = {
    "policy": NETWORK_POLICY_LOGITS,
    "wdl": NETWORK_WDL,
    "value": NETWORK_VALUE,
    "mlh": NETWORK_MLH,
}


class LczeroEvaluator:
    """Prepare chess positions, execute a model, and standardize its heads."""

    def __init__(
        self,
        model: LczeroModel,
        *,
        input_format: InputFormat = InputFormat.CLASSICAL_112,
    ):
        if not isinstance(model, LczeroModel):
            raise TypeError("model must be an LczeroModel.")
        if not isinstance(input_format, InputFormat):
            raise TypeError("input_format must be an InputFormat.")
        heads = tuple(key for key in model.out_keys if isinstance(key, str) and key != "_")
        if "policy" not in heads or any(head not in _NETWORK_KEYS for head in heads):
            raise ValueError("LczeroEvaluator requires a policy head and only supports policy, wdl, value, and mlh.")
        self._source_model = model
        self.model = TensorDictModule(
            model.module,
            in_keys=[INPUT_PLANES],
            out_keys=[_NETWORK_KEYS[head] for head in heads],
        )
        self.input_format = input_format

    @classmethod
    def from_path(cls, model_path: str, **kwargs) -> "LczeroEvaluator":
        """Load a model using the current Lczero model-format adapters."""
        return cls(LczeroModel.from_path(model_path), **kwargs)

    @property
    def device(self) -> torch.device:
        """Device of the wrapped neural module."""
        return self._source_model.device

    def prepare(self, boards: Sequence[chess.Board]) -> TensorDict:
        """Encode a non-empty position batch as the canonical input TensorDict."""
        resolved = _validate_boards(boards)
        planes = torch.stack([encode_input(board, input_format=self.input_format) for board in resolved]).to(
            self.device
        )
        masks = torch.stack([legal_mask(board) for board in resolved]).to(self.device)
        return TensorDict(
            {INPUT_PLANES: planes, INPUT_LEGAL_MASK: masks}, batch_size=[len(resolved)], device=self.device
        )

    def finish(self, boards: Sequence[chess.Board], tensors: TensorDictBase) -> EvaluationBatch:
        """Validate and standardize network heads without discarding added keys."""
        resolved = _validate_boards(boards)
        if not isinstance(tensors, TensorDictBase):
            raise TypeError("tensors must be a TensorDictBase.")
        if tensors.batch_dims != 1 or tensors.batch_size[0] != len(resolved):
            raise ValueError("TensorDict must have one batch dimension matching the board count.")
        _require_shape(tensors, INPUT_PLANES, (len(resolved), 112, 8, 8))
        _require_shape(tensors, INPUT_LEGAL_MASK, (len(resolved), POLICY_SIZE), dtype=torch.bool)
        _require_shape(tensors, NETWORK_POLICY_LOGITS, (len(resolved), POLICY_SIZE), finite=True)

        expected_masks = torch.stack([legal_mask(board) for board in resolved]).to(tensors.device)
        if not torch.equal(tensors[INPUT_LEGAL_MASK], expected_masks):
            raise ValueError("input/legal_mask does not match the supplied positions.")

        logits = tensors[NETWORK_POLICY_LOGITS]
        masks = tensors[INPUT_LEGAL_MASK]
        probabilities = torch.zeros_like(logits)
        for row in range(len(resolved)):
            if masks[row].any():
                probabilities[row, masks[row]] = torch.softmax(logits[row, masks[row]], dim=0)
        tensors[EVALUATION_POLICY] = probabilities

        leaves = set(tensors.keys(include_nested=True, leaves_only=True))
        if NETWORK_WDL in leaves:
            _require_shape(tensors, NETWORK_WDL, (len(resolved), 3), finite=True)
            wdl = tensors[NETWORK_WDL]
            if (
                (wdl < 0).any()
                or (wdl > 1).any()
                or not torch.allclose(
                    wdl.sum(-1), torch.ones(len(resolved), device=wdl.device, dtype=wdl.dtype), atol=1e-5
                )
            ):
                raise ValueError("network/wdl must contain probabilities in [0, 1] that sum to one.")

        if NETWORK_VALUE in leaves:
            value = _column(tensors[NETWORK_VALUE], len(resolved), "network/value")
            if not torch.isfinite(value).all() or (value.abs() > 1).any():
                raise ValueError("network/value must be finite and in [-1, 1].")
            tensors[NETWORK_VALUE] = value
            tensors[EVALUATION_VALUE] = value.clone()
        elif NETWORK_WDL in leaves:
            wdl = tensors[NETWORK_WDL]
            tensors[EVALUATION_VALUE] = (wdl[:, 0] - wdl[:, 2]).unsqueeze(-1)

        if NETWORK_MLH in leaves:
            mlh = _column(tensors[NETWORK_MLH], len(resolved), "network/mlh")
            if not torch.isfinite(mlh).all():
                raise ValueError("network/mlh must be finite.")
            tensors[NETWORK_MLH] = mlh
        return EvaluationBatch(resolved, tensors)

    def evaluate(self, boards: chess.Board | Sequence[chess.Board]) -> Evaluation | EvaluationBatch:
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
    if finite and not torch.isfinite(value).all():
        raise ValueError(f"TensorDict key {key!r} must be finite.")


def _column(value: torch.Tensor, batch_size: int, label: str) -> torch.Tensor:
    if tuple(value.shape) == (batch_size,):
        return value.unsqueeze(-1)
    if tuple(value.shape) == (batch_size, 1):
        return value
    raise ValueError(f"{label} must have shape [{batch_size}] or [{batch_size}, 1].")


__all__ = ["LczeroEvaluator"]
