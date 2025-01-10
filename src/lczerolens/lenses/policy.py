"""PolicyLens class for wrapping the LCZero models."""

import chess
import torch

from lczerolens.constants import (
    INVERTED_FROM_INDEX,
    INVERTED_TO_INDEX,
)
from lczerolens.model import LczeroModel, PolicyFlow
from lczerolens.lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@Lens.register("policy")
class PolicyLens(Lens):
    """
    Class for wrapping the LCZero models.
    """

    def is_compatible(self, model: LczeroModel) -> bool:
        return PolicyFlow.is_compatible(model)

    def analyse(
        self,
        board: chess.Board,
        model: LczeroModel,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the policy for a given board."""
        aggregate_topk = kwargs.pop("aggregate_topk", -1)

        policy_flow = PolicyFlow(model)
        (policy,) = policy_flow(board, **kwargs)
        return self.aggregate_policy(policy, aggregate_topk)

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
