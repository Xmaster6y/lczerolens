"""Preproces functions for chess games.

Classes
-------
Game
    A class for representing a game.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import chess


@dataclass
class Game:
    gameid: str
    moves: List[str]
    book_exit: Optional[int] = None


def dict_to_game(obj: Dict[str, str]) -> Game:
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
    return Game(
        gameid=obj["gameid"],
        moves=parsed_moves,
        book_exit=book_exit,
    )


def game_to_boards(
    game: Game,
    n_history: int = 0,
    skip_book_exit: bool = False,
    skip_first_n: int = 0,
    output_dict=True,
) -> List[Union[Dict[str, Any], chess.Board]]:
    working_board = chess.Board()
    if skip_first_n > 0 or (skip_book_exit and (game.book_exit is not None)):
        boards = []
    else:
        if output_dict:
            boards = [
                {
                    "fen": working_board.fen(),
                    "moves": [],
                    "gameid": game.gameid,
                }
            ]
        else:
            boards = [working_board.copy(stack=n_history)]

    for i, move in enumerate(game.moves[:-1]):  # skip the last move as it can be over
        working_board.push_san(move)
        if (i < skip_first_n) or (skip_book_exit and (game.book_exit is not None) and (i < game.book_exit)):
            continue
        if output_dict:
            save_board = working_board.copy(stack=n_history)
            boards.append(
                {
                    "fen": save_board.root().fen(),
                    "moves": [move.uci() for move in save_board.move_stack],
                    "gameid": game.gameid,
                }
            )
        else:
            boards.append(working_board.copy(stack=n_history))
    return boards
