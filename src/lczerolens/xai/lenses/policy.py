"""PolicyLens class for wrapping the LCZero models."""

from typing import Any, Callable, Dict, Optional

import chess
import torch
from torch.utils.data import DataLoader, Dataset

from lczerolens.encodings.constants import (
    INVERTED_FROM_INDEX,
    INVERTED_TO_INDEX,
)
from lczerolens.model.wrapper import ModelWrapper, PolicyFlow
from lczerolens.xai.lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@Lens.register("policy")
class PolicyLens(Lens):
    """
    Class for wrapping the LCZero models.
    """

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """Returns whether the lens is compatible with the model."""
        return PolicyFlow.is_compatible(wrapper.model)

    def analyse_board(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the policy for a given board."""
        policy_flow = PolicyFlow(wrapper.model)
        (policy,) = policy_flow.predict(board)
        return policy

    def analyse_dataset(
        self,
        dataset: Dataset,
        wrapper: ModelWrapper,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        save_to: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[int, Any]]:
        """Computes the statistics for a given board."""
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        policy_flow = PolicyFlow(wrapper.model)
        policies = {}
        for batch in loader:
            indices, boards = batch
            (batched_policies,) = policy_flow.predict(boards)
            for idx in indices:
                policies[idx] = batched_policies[idx]
        return policies

    @staticmethod
    def aggregate_policy(
        policy: torch.Tensor,
        aggregate_topk: int = -1,
    ):
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
            pickup_agg[square_index] = filtered_policy[INVERTED_FROM_INDEX[square]].sum()
            dropoff_agg[square_index] = filtered_policy[INVERTED_TO_INDEX[square]].sum()
        return pickup_agg, dropoff_agg
