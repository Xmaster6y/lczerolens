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
import tqdm
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
            with jsonlines.open(file_name) as reader:
                for obj in reader:
                    *pre, post = obj["moves"].split("{ Book exit }")
                    if pre:
                        if len(pre) > 1:
                            raise ValueError("More than one book exit")
                        (pre,) = pre
                        parsed_pre_moves = [
                            m for m in pre.split() if not m.endswith(".")
                        ]
                        book_exit = len(parsed_pre_moves)
                    else:
                        parsed_pre_moves = []
                        book_exit = None
                    parsed_moves = parsed_pre_moves + [
                        m for m in post.split() if not m.endswith(".")
                    ]

                    self.games.append(
                        Game(
                            gameid=obj["gameid"],
                            moves=parsed_moves,
                            book_exit=book_exit,
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
            with jsonlines.open(file_name) as reader:
                for obj in reader:
                    board = chess.Board(obj["fen"])
                    for move in obj["moves"]:
                        board.push_uci(move)

                    self.boards.append(board)
                    self.game_ids.append(obj["gameid"])

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx) -> Tuple[int, chess.Board]:
        return idx, self.boards[idx]

    def save(self, file_name: str, n_history: int = 0, indices=None):
        """Save the dataset to a file.

        Note
        ----
        As the board needs to be unpiled use the preprocess_game method.
        """
        print(f"[INFO] Saving boards to {file_name}")
        with jsonlines.open(file_name, "w") as writer:
            idx = 0
            for board, gameid in tqdm.tqdm(
                zip(self.boards, self.game_ids),
                total=len(self.boards),
                bar_format="{l_bar}{bar}",
            ):
                if indices is not None and idx not in indices:
                    idx += 1
                    continue
                idx += 1
                working_board = board.copy(stack=n_history)

                writer.write(
                    {
                        "fen": working_board.root().fen(),
                        "moves": [
                            move.uci() for move in working_board.move_stack
                        ],
                        "gameid": gameid,
                    }
                )

    @classmethod
    def from_game_dataset(
        cls,
        game_dataset: GameDataset,
        n_history: int = 0,
        skip_book_exit: bool = False,
        skip_first_n: int = 0,
    ):
        boards: List[chess.Board] = []
        game_ids: List[str] = []
        print("[INFO] Converting games to boards")
        for game in tqdm.tqdm(game_dataset.games, bar_format="{l_bar}{bar}"):
            new_boards, new_ids = cls.game_to_board_list(
                game, n_history, skip_book_exit, skip_first_n
            )
            boards.extend(new_boards)
            game_ids.extend(new_ids)

        return cls(boards=boards, game_ids=game_ids)

    @staticmethod
    def preprocess_game(
        game: Game,
        n_history: int = 0,
        skip_book_exit: bool = False,
        skip_first_n: int = 0,
    ) -> List[Dict[str, Any]]:
        working_board = chess.Board()
        if skip_first_n > 0 or (
            skip_book_exit and (game.book_exit is not None)
        ):
            boards = []
        else:
            boards = [
                {
                    "fen": working_board.fen(),
                    "moves": [],
                    "gameid": game.gameid,
                }
            ]
        for i, move in enumerate(
            game.moves[:-1]
        ):  # skip the last move as it can be over
            working_board.push_san(move)
            if (i < skip_first_n) or (
                skip_book_exit
                and (game.book_exit is not None)
                and (i < game.book_exit)
            ):
                continue
            save_board = working_board.copy(stack=n_history)
            boards.append(
                {
                    "fen": save_board.root().fen(),
                    "moves": [move.uci() for move in save_board.move_stack],
                    "gameid": game.gameid,
                }
            )
        return boards

    @staticmethod
    def game_to_board_list(
        game: Game,
        n_history: int = 0,
        skip_book_exit: bool = False,
        skip_first_n: int = 0,
    ) -> Tuple[List[chess.Board], List[str]]:
        working_board = chess.Board()
        if skip_first_n > 0 or (
            skip_book_exit and (game.book_exit is not None)
        ):
            boards = []
        else:
            boards = [working_board.copy(stack=n_history)]
        for i, move in enumerate(
            game.moves[:-1]
        ):  # skip the last move as it can be over
            working_board.push_san(move)
            if (i < skip_first_n) or (
                skip_book_exit
                and (game.book_exit is not None)
                and (i < game.book_exit)
            ):
                continue
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
