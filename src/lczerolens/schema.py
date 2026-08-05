"""Canonical TensorDict schema for Lczero evaluator execution."""

from __future__ import annotations

from typing import Final

from tensordict.utils import NestedKey


class LczeroKeys:
    """Stable nested keys used by the evaluator execution pipeline."""

    INPUT_PLANES: Final[NestedKey] = ("input", "planes")
    INPUT_LEGAL_MASK: Final[NestedKey] = ("input", "legal_mask")

    NETWORK_POLICY_LOGITS: Final[NestedKey] = ("network", "policy_logits")
    NETWORK_WDL: Final[NestedKey] = ("network", "wdl")
    NETWORK_VALUE: Final[NestedKey] = ("network", "value")
    NETWORK_MLH: Final[NestedKey] = ("network", "mlh")

    EVALUATION_POLICY: Final[NestedKey] = ("evaluation", "policy")
    EVALUATION_VALUE: Final[NestedKey] = ("evaluation", "value")


_NETWORK_HEAD_KEYS: Final[dict[str, NestedKey]] = {
    "policy": LczeroKeys.NETWORK_POLICY_LOGITS,
    "wdl": LczeroKeys.NETWORK_WDL,
    "value": LczeroKeys.NETWORK_VALUE,
    "mlh": LczeroKeys.NETWORK_MLH,
}


__all__ = ["LczeroKeys"]
