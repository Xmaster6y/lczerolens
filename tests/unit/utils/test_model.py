"""
Test for the model utils.
"""

import torch

from lczerolens import model_utils


class TestAggregatePolicy:
    def test_aggregate_policy_empty(self):
        """
        Test that the aggregate policy function works.
        """
        policy = torch.zeros(1858)
        pickup_agg, dropoff_agg = model_utils.aggregate_policy(policy)
        assert (pickup_agg == torch.zeros(64)).all()
        assert (dropoff_agg == torch.zeros(64)).all()

    def test_aggregate_policy_homogenous(self):
        """
        Test that the aggregate policy function works.
        """
        policy = torch.ones(1858)
        pickup_agg, dropoff_agg = model_utils.aggregate_policy(policy)
        assert pickup_agg.sum() == 1858
        assert dropoff_agg.sum() == 1858
        promotion_diff = torch.tensor([6, 9, 9, 9, 9, 9, 9, 6])
        agg_diff = pickup_agg - dropoff_agg
        assert (agg_diff[:48] == 0).all()
        assert (agg_diff[48:56] == promotion_diff).all()
        assert (agg_diff[56:] == -promotion_diff).all()
