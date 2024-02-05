"""
Dataset class for lczero models.
"""

from typing import List

import chess
import jsonlines
import torch
from torch.utils.data import Dataset

from .generate import Game


class GameDataset(Dataset):
    def __init__(
        self,
        file_name: str,
    ):
        self.games: List[Game] = []
        with jsonlines.open(file_name) as reader:
            offset = 0
            for obj in reader:
                parsed_moves = [
                    m for m in obj["moves"].split() if not m.endswith(".")
                ]
                self.games.append(
                    Game(
                        offset=offset, gameid=obj["gameid"], moves=parsed_moves
                    )
                )
                offset += len(parsed_moves) + 1
        self.device = torch.device("cpu")

    def __len__(self):
        last_game = self.games[-1]
        return last_game.offset + len(last_game.moves) + 1

    def _search_game(self, idx: int, inf_sup=None) -> int:
        if idx >= self.__len__():
            raise IndexError
        elif idx < 0:
            raise IndexError
        if inf_sup is None:
            inf = 0
            sup = len(self.games) - 1
        else:
            inf, sup = inf_sup
        if sup - inf <= 1:
            return inf
        mid = (inf + sup) // 2
        if idx >= self.games[mid].offset:
            return self._search_game(idx, (mid, sup))
        else:
            return self._search_game(idx, (inf, mid))

    def __getitem__(self, idx) -> chess.Board:
        game_idx = self._search_game(idx)
        game = self.games[game_idx]
        board = chess.Board()
        for move in game.moves[: idx - game.offset]:
            board.push_san(move)
        return board
