"""
Test the dataset module.
"""

from lczerolens import BoardDataset, GameDataset


class TestGameDataset:
    def test_access(self, game_dataset_10: GameDataset):
        assert len(game_dataset_10.games) == 10
        assert len(game_dataset_10[0].moves) == 129


class TestBoardDataset:
    def test_conversion(self, game_dataset_10: GameDataset):
        board_dataset = BoardDataset.from_game_dataset(game_dataset_10)
        assert len(board_dataset.boards) == 1169
        assert len(board_dataset[0].move_stack) == 0
