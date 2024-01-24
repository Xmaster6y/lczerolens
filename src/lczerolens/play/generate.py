from .search import SearchAlgorithm


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
