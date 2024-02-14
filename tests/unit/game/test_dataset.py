"""
Test the dataset module.
"""

import chess

from lczerolens.game import GameDataset


class TestAcces:
    def test_access(self, game_dataset_10: GameDataset):
        assert len(game_dataset_10.games) == 10
        assert game_dataset_10[0] == chess.Board()

    def test_offset_access(self, game_dataset_10: GameDataset):
        assert game_dataset_10.games[0].offset == 0
        assert len(game_dataset_10.games[0].moves) == 129
        assert game_dataset_10.games[1].offset == 130
        assert game_dataset_10[130] == chess.Board()

    def test_cache_access(self, game_dataset_10: GameDataset):
        board = chess.Board()
        assert game_dataset_10[130] == board
        assert game_dataset_10[130] == board
        board.push_san("e3")
        assert game_dataset_10[131] == board
        assert game_dataset_10.cache == (131, 1, board)
        board.pop()
        assert game_dataset_10[130] == board
        assert game_dataset_10.cache == (130, 1, board)