"""Dataset class for lczero models.

Classes
-------
GameDataset
    A class for representing a dataset of games.
BoardDataset
    A class for representing a dataset of boards.
IterableBoardDataset
    A class for representing an iterable dataset of boards.
"""

from typing import Any, Dict, List, Optional

import chess
import jsonlines
import torch
from torch.utils.data import Dataset, IterableDataset

from lczerolens.utils import board as board_utils

from .generate import Game


class GameDataset(Dataset):
    """A class for representing a dataset of games.

    Attributes
    ----------
    games : List[Game]
        The list of games.
    """

    def __init__(
        self,
        file_name: Optional[str],
    ):
        self.games: List[Game] = []
        if file_name is not None:
            with jsonlines.open(file_name) as reader:
                for obj in reader:
                    parsed_moves = [
                        m for m in obj["moves"].split() if not m.endswith(".")
                    ]
                    self.games.append(
                        Game(
                            gameid=obj["gameid"],
                            moves=parsed_moves,
                        )
                    )

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx) -> Game:
        return self.games[idx]


class BoardDataset(Dataset):
    """A class for representing a dataset of boards.

    Attributes
    ----------
    boards : List[chess.Board]
        The list of boards.

    Methods
    -------
    from_game_dataset(
        game_dataset: GameDataset,
        n_history: int = 0,
    )
        Creates a board dataset from a game dataset.
    preprocess_game(
        game: Game,
        n_history: int = 0,
    ) -> List[Dict[str, Any]]
        Preprocesses a game into a list of boards.
    collate_fn_list(batch: List) -> List
        Collate function for lists.
    collate_fn_tensor(batch: List) -> torch.Tensor
        Collate function for tensors.
    """

    def __init__(
        self,
        file_name: Optional[str],
    ):
        self.boards = []
        if file_name is not None:
            with jsonlines.open(file_name) as reader:
                for obj in reader:
                    board = chess.Board(obj["fen"])
                    for move in obj["moves"]:
                        board.push(chess.Move.from_uci(move))
                    self.boards.append(board)

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx) -> chess.Board:
        return self.boards[idx]

    @classmethod
    def from_game_dataset(
        cls,
        game_dataset: GameDataset,
        n_history: int = 0,
    ):
        instance = cls(None)
        for game in game_dataset.games:
            instance.boards.extend(cls.preprocess_game(game, n_history))

    @staticmethod
    def preprocess_game(
        game: Game,
        n_history: int = 0,
    ) -> List[Dict[str, Any]]:
        board = chess.Board()
        boards = [
            {
                "fen": board.fen(),
                "moves": [],
            }
        ]
        for i in range(len(game.moves)):
            if i >= n_history:
                board.push(chess.Move.from_uci(game.moves[i - n_history]))
            boards.append(
                {
                    "fen": board.fen(),
                    "moves": game.moves[i - n_history : i],
                }
            )
        return boards

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


class IterableBoardDataset(IterableDataset):
    """A class for representing an iterable dataset of boards.

    Note
    ----
    Usefull for large datasets that do not fit into memory.
    """

    def __init__(
        self,
        file_path,
        n_parts=1,
    ):
        pass

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return iter(jsonlines.open(self.file_path))
        else:
            worker_id = worker_info.id
            return iter(
                jsonlines.open(f"{self.base_name}{worker_id}.{self.ext}")
            )

    def __len__(self):
        return self.n_lines
