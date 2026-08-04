"""Private Lczero position and policy codecs.

The public evaluator accepts :class:`chess.Board` directly.  These stateless
helpers own the Lczero-specific tensor and policy vocabulary mapping without
introducing another mutable board type.
"""

from .input import InputFormat, encode_input
from .policy import POLICY_SIZE, decode_move, encode_move, legal_indices, legal_mask

__all__ = [
    "InputFormat",
    "POLICY_SIZE",
    "decode_move",
    "encode_input",
    "encode_move",
    "legal_indices",
    "legal_mask",
]
