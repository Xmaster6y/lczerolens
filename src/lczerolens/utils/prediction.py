"""
Utils for the model module.
"""

from typing import List

import chess
import torch
from tensordict import TensorDict

from . import board as board_utils
from .constants import INVERTED_FROM_INDEX, INVERTED_TO_INDEX


def aggregate_policy(policy: torch.Tensor, aggregate_topk: int = -1):
    """
    Aggregate the policy for a given board.
    """
    pickup_agg = torch.zeros(64)
    dropoff_agg = torch.zeros(64)
    if aggregate_topk > 0:
        filtered_policy = torch.zeros(1858)
        topk = torch.topk(policy, aggregate_topk)
        filtered_policy[topk.indices] = topk.values
    else:
        filtered_policy = policy
    for square_index in range(64):
        square = chess.SQUARE_NAMES[square_index]
        pickup_agg[square_index] = filtered_policy[
            INVERTED_FROM_INDEX[square]
        ].sum()
        dropoff_agg[square_index] = filtered_policy[
            INVERTED_TO_INDEX[square]
        ].sum()
    return pickup_agg, dropoff_agg


def compute_move_prediction(
    model: torch.nn.Module,
    board_list: List[chess.Board],
    with_grad: bool = False,
    input_requires_grad: bool = False,
    return_input: bool = False,
) -> TensorDict:
    """
    Compute the move prediction for a list of boards.
    """
    tensor_list = [
        board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
        for board in board_list
    ]
    batched_tensor = torch.cat(tensor_list, dim=0)
    if input_requires_grad:
        batched_tensor.requires_grad = True

    with torch.set_grad_enabled(with_grad):
        out = model(batched_tensor)
        if len(out) == 2:
            policy, other = out
            if other.shape[1] == 3:
                outcome_probs = other
                value = torch.zeros((outcome_probs.shape[0], 1))
            elif other.shape[1] == 1:
                value = other
                outcome_probs = torch.zeros((value.shape[0], 3))
            else:
                raise ValueError(f"Unexpected output shape {other.shape}.")
        else:
            policy, outcome_probs, value = out
    out_d = {
        "policy": policy,
        "outcome_probs": outcome_probs,
        "value": value,
    }
    if return_input:
        out_d["input"] = batched_tensor
    return TensorDict(
        out_d,
        batch_size=batched_tensor.shape[0],
    )
