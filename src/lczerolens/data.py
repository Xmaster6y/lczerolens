"""
Data classes for chess game, board, and puzzle.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Iterable, Tuple

import torch
import chess
from itertools import tee, chain

from lczerolens.sampling import Sampler
from lczerolens.concepts import Concept
from lczerolens.board import LczeroBoard


@dataclass
class GameData:
    """Data class representing a chess game with moves and metadata.

    Attributes
    ----------
    gameid : str
        Unique identifier for the game.
    moves : List[str]
        List of moves in the game in standard algebraic notation.
    book_exit : Optional[int]
        Move number where the game exits book theory, if applicable.
    """

    gameid: str
    moves: List[str]
    book_exit: Optional[int] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, str]) -> "GameData":
        """Create a GameData instance from a dictionary.

        Parameters
        ----------
        obj : Dict[str, str]
            Dictionary containing game data with 'gameid' and 'moves' keys.
            The 'moves' value should contain moves separated by spaces,
            with "{ Book exit }" marking the book exit point.

        Returns
        -------
        GameData
            A new GameData instance.

        Raises
        ------
        ValueError
            If required keys are missing or if there are multiple book exit markers.
        """
        if "moves" not in obj:
            raise ValueError("The dict should contain `moves`.")
        if "gameid" not in obj:
            raise ValueError("The dict should contain `gameid`.")
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
        concept: Optional[Concept] = None,
    ) -> List[Union[Dict[str, Any], LczeroBoard]]:
        """Convert the game to a list of board positions.

        Parameters
        ----------
        n_history : int, default=0
            Number of previous moves to include in the board's move stack.
        skip_book_exit : bool, default=False
            Whether to skip positions before book exit.
        skip_first_n : int, default=0
            Number of initial positions to skip.
        output_dict : bool, default=True
            If True, return dictionaries with board information.
            If False, return LczeroBoard objects directly.
        concept : Optional[Concept], default=None
            Concept to compute labels for each board position.

        Returns
        -------
        List[Union[Dict[str, Any], LczeroBoard]]
            List of board representations, either as dictionaries or LczeroBoard objects.
        """
        working_board = LczeroBoard()
        label = concept.compute_label(working_board) if concept is not None else None

        if skip_first_n > 0 or (skip_book_exit and (self.book_exit is not None)):
            boards = []
        elif output_dict:
            boards = [
                {
                    "fen": working_board.fen(),
                    "moves": [],
                    "gameid": self.gameid,
                    "label": label,
                }
            ]
        else:
            boards = [working_board.copy(stack=n_history)]

        for i, move in enumerate(self.moves[:-1]):  # skip the last move as it can be over
            working_board.push_san(move)
            label = concept.compute_label(working_board) if concept is not None else None
            if (i < skip_first_n) or (skip_book_exit and (self.book_exit is not None) and (i < self.book_exit)):
                continue
            if output_dict:
                save_board = working_board.copy(stack=n_history)
                boards.append(
                    {
                        "fen": save_board.root().fen(),
                        "moves": [move.uci() for move in save_board.move_stack],
                        "gameid": self.gameid,
                        "label": label,
                    }
                )
            else:
                boards.append(working_board.copy(stack=n_history))
        return boards

    @staticmethod
    def collate_fn(batch):
        """Collate function for batching GameData objects.

        Parameters
        ----------
        batch : List[Dict[str, str]]
            List of dictionaries containing game data.

        Returns
        -------
        List[GameData]
            List of GameData instances.
        """
        return [GameData.from_dict(element) for element in batch]

    @staticmethod
    def get_dataset_features():
        """Returns the features for the game dataset.

        Returns
        -------
        Features
            Dataset features configuration for the HuggingFace datasets library.

        Raises
        ------
        ImportError
            If the datasets library is not available.
        """
        try:
            from datasets import Features, Value, Sequence
        except ImportError as e:
            raise ImportError("datasets is required to get the dataset features.") from e
        return Features(
            {
                "gameid": Value("string"),
                "moves": Sequence(Value("string")),
            }
        )


@dataclass
class BoardData:
    """Data class representing a single chess board position with metadata.

    Attributes
    ----------
    gameid : str
        Unique identifier for the game this board belongs to.
    moves : List[str]
        List of moves that led to this board position.
    fen : str
        FEN string representation of the board position.
    label : Optional[Any]
        Optional label for the board position (e.g., concept-based label).
    """

    gameid: str
    moves: List[str]
    fen: str
    label: Optional[Any] = None

    @staticmethod
    def get_dataset_features(concept: Optional[Concept] = None):
        """Returns the features for the board dataset.

        Parameters
        ----------
        concept : Optional[Concept], default=None
            Concept to determine the label feature type.

        Returns
        -------
        Features
            Dataset features configuration for the HuggingFace datasets library.

        Raises
        ------
        ImportError
            If the datasets library is not available.
        """
        try:
            from datasets import Features, Value, Sequence
        except ImportError as e:
            raise ImportError("datasets is required to get the dataset features.") from e
        concept_feature = concept.get_dataset_feature() if concept is not None else Value("null")
        return Features(
            {
                "gameid": Value("string"),
                "moves": Sequence(Value("string")),
                "fen": Value("string"),
                "label": concept_feature,
            }
        )

    @staticmethod
    def board_collate_fn(batch):
        """Collate function for creating LczeroBoard objects from batch data.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            List of dictionaries containing board data with 'fen' and 'moves' keys.

        Returns
        -------
        List[LczeroBoard]
            List of LczeroBoard instances.
        """
        boards = []
        for element in batch:
            board = LczeroBoard(element["fen"])
            for move in element["moves"]:
                board.push_san(move)
            boards.append(board)
        return boards

    @staticmethod
    def concept_collate_fn(batch, concept: Optional[Concept] = None):
        """Collate function for concept-based analysis with labels.

        Parameters
        ----------
        batch : List[Dict[str, Any]]
            List of dictionaries containing board data and labels.
        concept : Optional[Concept], default=None
            Concept to determine the label feature type.

        Returns
        -------
        Tuple[List[LczeroBoard], List[Any], List[Dict[str, Any]]]
            Tuple containing boards, labels, and original batch data.
        """
        boards = []
        labels = []
        for element in batch:
            board = LczeroBoard(element["fen"])
            for move in element["moves"]:
                board.push_san(move)
            boards.append(board)
            if concept is not None:
                labels.append(concept.compute_label(board))
            else:
                labels.append(element["label"])
        return boards, labels

    @staticmethod
    def concept_init_grad(output, infos):
        """Initialize gradients for concept-based analysis.

        Parameters
        ----------
        output : torch.Tensor
            Model output tensor.
        infos : Tuple
            Tuple containing labels and other information.

        Returns
        -------
        torch.Tensor
            Gradient tensor initialized for concept analysis.
        """
        labels = infos[0]
        rel = torch.zeros_like(output)
        for i in range(rel.shape[0]):
            rel[i, labels[i]] = output[i, labels[i]]
        return rel


@dataclass
class PuzzleData:
    """Data class representing a chess puzzle with solution and metadata.

    Attributes
    ----------
    puzzle_id : str
        Unique identifier for the puzzle.
    fen : str
        FEN string representation of the puzzle's starting position.
    initial_move : chess.Move
        The first move that must be played to solve the puzzle.
    moves : List[chess.Move]
        List of moves in the solution sequence.
    rating : int
        Puzzle rating indicating difficulty.
    rating_deviation : int
        Statistical deviation of the rating.
    popularity : int
        Popularity score of the puzzle.
    nb_plays : int
        Number of times the puzzle has been played.
    themes : List[str]
        List of chess themes/tactics present in the puzzle.
    game_url : str
        URL to the original game where the puzzle occurred.
    opening_tags : List[str]
        List of opening tags associated with the puzzle.
    """

    puzzle_id: str
    fen: str
    initial_move: chess.Move
    moves: List[chess.Move]
    rating: int
    rating_deviation: int
    popularity: int
    nb_plays: int
    themes: List[str]
    game_url: str
    opening_tags: List[str]

    @classmethod
    def from_dict(cls, obj: Dict[str, Union[str, int, None]]) -> "PuzzleData":
        """Create a PuzzleData instance from a dictionary.

        Parameters
        ----------
        obj : Dict[str, Union[str, int, None]]
            Dictionary containing puzzle data with keys matching the class attributes.
            'Moves' should be a space-separated string of UCI moves.
            'Themes' and 'OpeningTags' should be space-separated strings.

        Returns
        -------
        PuzzleData
            A new PuzzleData instance.
        """
        uci_moves = obj["Moves"].split()
        moves = [chess.Move.from_uci(uci_move) for uci_move in uci_moves]
        return cls(
            puzzle_id=obj["PuzzleId"],
            fen=obj["FEN"],
            initial_move=moves[0],
            moves=moves[1:],
            rating=obj["Rating"],
            rating_deviation=obj["RatingDeviation"],
            popularity=obj["Popularity"],
            nb_plays=obj["NbPlays"],
            themes=obj["Themes"].split() if obj["Themes"] is not None else [],
            game_url=obj["GameUrl"],
            opening_tags=obj["OpeningTags"].split() if obj["OpeningTags"] is not None else [],
        )

    def __len__(self) -> int:
        """Return the number of moves in the puzzle solution.

        Returns
        -------
        int
            Number of moves in the solution sequence.
        """
        return len(self.moves)

    @property
    def initial_board(self) -> LczeroBoard:
        """Get the board position after the initial move is played.

        Returns
        -------
        LczeroBoard
            Board position after the initial move.
        """
        board = LczeroBoard(self.fen)
        board.push(self.initial_move)
        return board

    def board_move_generator(self, all_moves: bool = False) -> Iterable[Tuple[LczeroBoard, chess.Move]]:
        """Generate board positions and moves for the puzzle solution.

        Parameters
        ----------
        all_moves : bool, default=False
            If True, yield all moves. If False, only yield moves for the initial player's turn.

        Yields
        ------
        Tuple[LczeroBoard, chess.Move]
            Board position and the move to be played.
        """
        board = self.initial_board
        initial_turn = board.turn
        for move in self.moves:
            if not all_moves and board.turn != initial_turn:
                board.push(move)
                continue
            yield board.copy(), move
            board.push(move)

    @classmethod
    def evaluate_multiple(
        cls,
        puzzles: Iterable["PuzzleData"],
        sampler: Sampler,
        all_moves: bool = False,
        compute_metrics: bool = True,
        **kwargs,
    ) -> Union[Iterable[Dict[str, float]], Iterable[Tuple[torch.Tensor, torch.Tensor, chess.Move]]]:
        """Evaluate multiple puzzles using a sampler.

        Parameters
        ----------
        puzzles : Iterable[PuzzleData]
            Collection of puzzles to evaluate.
        sampler : Sampler
            Sampler to use for move prediction and evaluation.
        all_moves : bool, default=False
            Whether to evaluate all moves or only initial player moves.
        compute_metrics : bool, default=True
            Whether to compute and return metrics or raw evaluation data.
        **kwargs
            Additional arguments to pass to the sampler.

        Returns
        -------
        Union[Iterable[Dict[str, float]], Iterable[Tuple[torch.Tensor, torch.Tensor, chess.Move]]]
            Either metrics for each puzzle or raw evaluation data.
        """
        metric_puzzles, board_move_puzzles = tee(puzzles)
        board_move_generator = chain.from_iterable(
            puzzle.board_move_generator(all_moves) for puzzle in board_move_puzzles
        )

        def board_generator():
            for board, _ in board_move_generator:
                yield board

        util_boards, move_boards = tee(board_generator())

        def metric_inputs_generator():
            util_gen = sampler.get_utilities(util_boards, **kwargs)
            for board, (utility, legal_indices, _) in zip(move_boards, util_gen):
                predicted_move = sampler.choose_move(board, utility, legal_indices)
                yield utility, legal_indices, predicted_move

        if compute_metrics:
            return cls.compute_metrics(metric_puzzles, metric_inputs_generator(), all_moves=all_moves)
        else:
            return metric_inputs_generator()

    def evaluate(self, sampler: Sampler, all_moves: bool = False, **kwargs) -> Tuple[float, Optional[float]]:
        """Evaluate this single puzzle using a sampler.

        Parameters
        ----------
        sampler : Sampler
            Sampler to use for move prediction and evaluation.
        all_moves : bool, default=False
            Whether to evaluate all moves or only initial player moves.
        **kwargs
            Additional arguments to pass to the sampler.

        Returns
        -------
        Tuple[float, Optional[float]]
            Tuple containing score and perplexity metrics.
        """
        return next(iter(self.evaluate_multiple([self], sampler, all_moves, **kwargs)))

    @staticmethod
    def compute_metrics(
        puzzles: Iterable["PuzzleData"],
        inputs: Iterable[Tuple[torch.Tensor, torch.Tensor, chess.Move]],
        all_moves: bool = False,
    ) -> Iterable[Dict[str, float]]:
        """Compute evaluation metrics for a collection of puzzles.

        Parameters
        ----------
        puzzles : Iterable[PuzzleData]
            Collection of puzzles to evaluate.
        inputs : Iterable[Tuple[torch.Tensor, torch.Tensor, chess.Move]]
            Iterator providing utility tensors, legal indices, and predicted moves.
        all_moves : bool, default=False
            Whether all moves were evaluated or only initial player moves.

        Yields
        ------
        Dict[str, float]
            Dictionary containing score, perplexity, and normalized perplexity metrics.
        """
        iter_inputs = iter(inputs)
        for puzzle in puzzles:
            total = len(puzzle) if all_moves else (len(puzzle) + 1) // 2
            metrics = {"score": 0.0, "perplexity": 1.0, "normalized_perplexity": 1.0}
            for board, move in puzzle.board_move_generator(all_moves=all_moves):
                utility, legal_indices, predicted_move = next(iter_inputs)
                index = LczeroBoard.encode_move(move, board.turn)
                probs = torch.softmax(utility, dim=0)
                move_prob = probs[legal_indices == index].item()
                metrics["perplexity"] *= move_prob ** (-1 / total)
                metrics["normalized_perplexity"] *= (len(legal_indices) * move_prob) ** (-1 / total)
                if predicted_move == move:
                    metrics["score"] += 1
            metrics["score"] /= total
            yield metrics

    def _repr_svg_(self) -> str:
        """Return SVG representation of the puzzle's initial board.

        Returns
        -------
        str
            SVG string representation of the board.
        """
        return self.initial_board._repr_svg_()

    @staticmethod
    def get_dataset_features():
        """Returns the features for the puzzle dataset.

        Returns
        -------
        Features
            Dataset features configuration for the HuggingFace datasets library.

        Raises
        ------
        ImportError
            If the datasets library is not available.
        """
        try:
            from datasets import Features, Value
        except ImportError as e:
            raise ImportError("datasets is required to get the dataset features.") from e

        return Features(
            {
                "PuzzleId": Value("string"),
                "FEN": Value("string"),
                "Moves": Value("string"),
                "Rating": Value("int64"),
                "RatingDeviation": Value("int64"),
                "Popularity": Value("int64"),
                "NbPlays": Value("int64"),
                "Themes": Value("string"),
                "GameUrl": Value("string"),
                "OpeningTags": Value("string"),
            }
        )
