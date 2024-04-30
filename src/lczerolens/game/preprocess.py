"""Preproces functions for chess games.
"""

from .generator import Game


def dict_to_game(obj: dict) -> Game:
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
    parsed_moves = parsed_pre_moves + [
        m for m in post.split() if not m.endswith(".")
    ]
    return Game(
        gameid=obj["gameid"],
        moves=parsed_moves,
        book_exit=book_exit,
    )
