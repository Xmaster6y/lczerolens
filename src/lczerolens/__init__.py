"""Main module for the lczerolens package.
"""

__version__ = "0.1.2"


from .game import BoardDataset, GameDataset, ModelWrapper
from .utils import board as board_utils
from .utils import move as move_utils
from .xai import Lens
