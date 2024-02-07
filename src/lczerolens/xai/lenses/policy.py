"""
Compute LRP heatmap for a given model and input.
"""

from typing import Any, Dict

import chess
import torch
from torch.utils.data import DataLoader

from lczerolens.adapt.wrapper import ModelWrapper, PolicyFlow
from lczerolens.game.dataset import GameDataset
from lczerolens.utils import move as move_utils
from lczerolens.utils.constants import INVERTED_FROM_INDEX, INVERTED_TO_INDEX
from lczerolens.xai.lens import Lens

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyLens(Lens):
    """
    Class for wrapping the LCZero models.
    """

    def is_compatible(self, wrapper: ModelWrapper) -> bool:
        """
        Returns whether the lens is compatible with the model.
        """
        if hasattr(wrapper.model, "policy"):
            return True
        else:
            return False

    def compute_heatmap(
        self,
        board: chess.Board,
        wrapper: ModelWrapper,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the LRP heatmap for a given model and input.
        """
        aggregate_topk = kwargs.get("aggregate_topk", -1)
        policy = kwargs.get("policy", None)
        if policy is None:
            policy = wrapper.predict(board)["policy"]
        return self.aggregate_policy(policy, aggregate_topk=aggregate_topk)

    def compute_statistics(
        self,
        dataset: GameDataset,
        wrapper: ModelWrapper,
        batch_size: int,
        **kwargs,
    ) -> dict:
        """
        Computes the statistics for a given board.
        """
        policy_flow = PolicyFlow(wrapper.model)
        statistics: Dict[str, Dict[int, Any]] = {
            "mean_legal_logits": {},
            "mean_illegal_logits": {},
        }
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=GameDataset.collate_fn_board_list,
        )
        for batch in dataloader:
            policy = policy_flow.predict(batch)
            for i, board in enumerate(batch):
                legal_moves = [
                    move_utils.encode_move(move, (board.turn, not board.turn))
                    for move in board.legal_moves
                ]
                legal_mask = torch.Tensor(
                    [move in legal_moves for move in range(1858)]
                ).bool()
                legal_mean = policy[i][legal_mask].mean().item()
                illegal_mean = policy[i][~legal_mask].mean().item()
                move_idx = len(board.move_stack)
                if move_idx not in statistics["mean_legal_logits"]:
                    statistics["mean_legal_logits"][move_idx] = [legal_mean]
                    statistics["mean_illegal_logits"][move_idx] = [
                        illegal_mean
                    ]
                else:
                    statistics["mean_legal_logits"][move_idx].append(
                        legal_mean
                    )
                    statistics["mean_illegal_logits"][move_idx].append(
                        illegal_mean
                    )
        return statistics

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
            pickup_agg[square_index] = filtered_policy[
                INVERTED_FROM_INDEX[square]
            ].sum()
            dropoff_agg[square_index] = filtered_policy[
                INVERTED_TO_INDEX[square]
            ].sum()
        return pickup_agg, dropoff_agg
