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

from typing import Any, Dict, List, Optional, Tuple

import chess
import jsonlines
import torch
from torch.utils.data import Dataset

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
        file_name: Optional[str] = None,
        games: Optional[List[Game]] = None,
    ):
        if games is None and file_name is None:
            raise ValueError("Either games or file_name must be provided")
        elif games is not None:
            self.games = games
        else:
            self.games = []
            if file_name is not None:
                with jsonlines.open(file_name) as reader:
                    for obj in reader:
                        parsed_moves = [
                            m
                            for m in obj["moves"].split()
                            if not m.endswith(".")
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

    def save(self, file_name: str):
        with jsonlines.open(file_name, "w") as writer:
            for game in self.games:
                writer.write(
                    {
                        "gameid": game.gameid,
                        "moves": " ".join(game.moves),
                    }
                )


class BoardDataset(Dataset):
    """A class for representing a dataset of boards.

    Attributes
    ----------
    boards : List[chess.Board]
        The list of boards.
    """

    def __init__(
        self,
        file_name: Optional[str] = None,
        boards: Optional[List[chess.Board]] = None,
        game_ids: Optional[List[str]] = None,
    ):
        if boards is not None and file_name is not None:
            raise ValueError("Either boards or file_name must be provided")
        elif boards is not None:
            self.boards = boards
            if game_ids is not None:
                self.game_ids = game_ids
            else:
                self.game_ids = ["none"] * len(boards)
        else:
            self.boards = []
            self.game_ids = []
            if file_name is not None:
                with jsonlines.open(file_name) as reader:
                    for obj in reader:
                        board = chess.Board(obj["fen"])
                        for move in obj["moves"]:
                            board.push_san(move)

                        self.boards.append(board)
                        self.game_ids.append(obj["gameid"])

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx) -> Tuple[int, chess.Board]:
        return idx, self.boards[idx]

    def save(self, file_name: str):
        with jsonlines.open(file_name, "w") as writer:
            for board, gameid in zip(self.boards, self.game_ids):
                writer.write(
                    {
                        "fen": board.fen(),
                        "moves": [move.uci() for move in board.move_stack],
                        "gameid": gameid,
                    }
                )

    @classmethod
    def from_game_dataset(
        cls,
        game_dataset: GameDataset,
        n_history: int = 0,
    ):
        boards: List[chess.Board] = []
        game_ids: List[str] = []
        for game in game_dataset.games:
            new_boards, new_ids = cls.game_to_board_list(game, n_history)
            boards.extend(new_boards)
            game_ids.extend(new_ids)

        return cls(boards=boards, game_ids=game_ids)

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
                "gameid": game.gameid,
            }
        ]
        for i in range(len(game.moves)):
            if i >= n_history:
                board.push_san(game.moves[i - n_history])
            boards.append(
                {
                    "fen": board.fen(),
                    "moves": game.moves[i - n_history : i],
                    "gameid": game.gameid,
                }
            )
        return boards

    @staticmethod
    def game_to_board_list(
        game: Game,
        n_history: int = 0,
    ) -> Tuple[List[chess.Board], List[str]]:
        working_board = chess.Board()
        boards = [working_board.copy(stack=n_history)]
        for move in game.moves:
            working_board.push_san(move)
            boards.append(working_board.copy(stack=n_history))
        return boards, [game.gameid] * len(boards)

    @staticmethod
    def collate_fn_tuple(batch):
        indices, boards = zip(*batch)
        return indices, boards

    @staticmethod
    def collate_fn_tensor(batch):
        tensor_list = [
            board_utils.board_to_input_tensor(board).unsqueeze(0)
            for board in batch
        ]
        batched_tensor = torch.cat(tensor_list, dim=0)
        return batched_tensor
