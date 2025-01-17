"""Preproces functions for chess games."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from datasets import Features, Value, Sequence

from lczerolens.board import LczeroBoard

GAME_DATASET_FEATURES = Features(
    {
        "gameid": Value("string"),
        "moves": Value("string"),
    }
)

BOARD_DATASET_FEATURES = Features(
    {
        "gameid": Value("string"),
        "moves": Sequence(Value("string")),
        "fen": Value("string"),
    }
)


@dataclass
class Game:
    gameid: str
    moves: List[str]
    book_exit: Optional[int] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, str]) -> "Game":
        if "moves" not in obj:
            ValueError("The dict should contain `moves`.")
        if "gameid" not in obj:
            ValueError("The dict should contain `gameid`.")
        *pre, post = obj["moves"].split("{ Book exit }")
        if pre:
            if len(pre) > 1:
                raise ValueError("More than one book exit")
            (pre,) = pre
            parsed_pre_moves = [m for m in pre.split() if not m.endswith(".")]
            book_exit = len(parsed_pre_moves)
        else:
            parsed_pre_moves = []
            book_exit = None
        parsed_moves = parsed_pre_moves + [m for m in post.split() if not m.endswith(".")]
        return cls(
            gameid=obj["gameid"],
            moves=parsed_moves,
            book_exit=book_exit,
        )

    def to_boards(
        self,
        n_history: int = 0,
        skip_book_exit: bool = False,
        skip_first_n: int = 0,
        output_dict=True,
    ) -> List[Union[Dict[str, Any], LczeroBoard]]:
        working_board = LczeroBoard()
        if skip_first_n > 0 or (skip_book_exit and (self.book_exit is not None)):
            boards = []
        else:
            if output_dict:
                boards = [
                    {
                        "fen": working_board.fen(),
                        "moves": [],
                        "gameid": self.gameid,
                    }
                ]
            else:
                boards = [working_board.copy(stack=n_history)]

        for i, move in enumerate(self.moves[:-1]):  # skip the last move as it can be over
            working_board.push_san(move)
            if (i < skip_first_n) or (skip_book_exit and (self.book_exit is not None) and (i < self.book_exit)):
                continue
            if output_dict:
                save_board = working_board.copy(stack=n_history)
                boards.append(
                    {
                        "fen": save_board.root().fen(),
                        "moves": [move.uci() for move in save_board.move_stack],
                        "gameid": self.gameid,
                    }
                )
            else:
                boards.append(working_board.copy(stack=n_history))
        return boards

    @staticmethod
    def board_collate_fn(batch):
        boards = []
        for element in batch:
            board = LczeroBoard(element["fen"])
            for move in element["moves"]:
                board.push_san(move)
            boards.append(board)
        return boards, {}
