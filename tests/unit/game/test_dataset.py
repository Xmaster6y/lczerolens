"""
Test the dataset module.
"""

from lczerolens.game import GameDataset


class TestGameDataset:
    def test_access(self, game_dataset_10: GameDataset):
        assert len(game_dataset_10.games) == 10
        assert len(game_dataset_10[0].moves) == 129
