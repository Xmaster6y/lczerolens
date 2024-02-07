"""
Dataset class for lczero models.
"""

from typing import List, Optional, Tuple

import chess
import jsonlines
import torch
from torch.utils.data import Dataset

from lczerolens.utils import board as board_utils

from .generate import Game


class GameDataset(Dataset):
    def __init__(
        self,
        file_name: Optional[str],
    ):
        self.games: List[Game] = []
        if file_name is not None:
            with jsonlines.open(file_name) as reader:
                offset = 0
                for obj in reader:
                    parsed_moves = [
                        m for m in obj["moves"].split() if not m.endswith(".")
                    ]
                    self.games.append(
                        Game(
                            offset=offset,
                            gameid=obj["gameid"],
                            moves=parsed_moves,
                        )
                    )
                    offset += len(parsed_moves) + 1
        self.device = torch.device("cpu")
        self.cache: Optional[Tuple[int, int, chess.Board]] = None

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
        if self.cache is not None:
            cache_idx, cache_game_idx, cache_board = self.cache
            idx_rescaled = idx - self.games[cache_game_idx].offset
            if idx_rescaled >= 0 and idx_rescaled < len(
                self.games[cache_game_idx].moves
            ):
                board = cache_board
                if cache_idx < idx:
                    for move in self.games[cache_game_idx].moves[
                        cache_idx
                        - self.games[cache_game_idx].offset : idx_rescaled
                    ]:
                        board.push_san(move)
                else:
                    for _ in range(idx, cache_idx):
                        board.pop()
                self.cache = (idx, cache_game_idx, board.copy())
                return board
        game_idx = self._search_game(idx)
        game = self.games[game_idx]
        board = chess.Board()
        for move in game.moves[: idx - game.offset]:
            board.push_san(move)
        self.cache = (idx, game_idx, board.copy())
        return board

    @staticmethod
    def collate_fn_list(batch):
        return batch

    @staticmethod
    def collate_fn_tensor(batch):
        tensor_list = [
            board_utils.board_to_tensor112x8x8(board).unsqueeze(0)
            for board in batch
        ]
        batched_tensor = torch.cat(tensor_list, dim=0)
        return batched_tensor
