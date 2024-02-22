"""Classes for generating games.

Classes
-------
Game
    A class for representing a game.
GameGenerator
    A class for generating games.
"""

from dataclasses import dataclass
from typing import List, Optional

from .search import SearchAlgorithm


@dataclass
class Game:
    gameid: str
    moves: List[str]
    book_exit: Optional[int] = None


class GameGenerator:
    """A class for generating games."""

    def __init__(self, white: SearchAlgorithm, black: SearchAlgorithm):
        """
        Initializes the game generator.
        """
        self.white = white
        self.black = black

    def play(self):
        """
        Plays a game.
        """
        raise NotImplementedError
