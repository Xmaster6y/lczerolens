"""
Data classes.
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
    gameid: str
    moves: List[str]
    book_exit: Optional[int] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, str]) -> "GameData":
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
    ) -> List[Union[Dict[str, Any], LczeroBoard]]:
        working_board = LczeroBoard()
        if skip_first_n > 0 or (skip_book_exit and (self.book_exit is not None)):
            boards = []
        elif output_dict:
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

    @staticmethod
    def get_dataset_features():
        """Returns the features for the game dataset."""
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
    gameid: str
    moves: List[str]
    fen: str
    label: Optional[Any] = None

    @staticmethod
    def get_dataset_features(concept: Optional[Concept] = None):
        """Returns the features for the board dataset."""
        try:
            from datasets import Features, Value, Sequence
        except ImportError as e:
            raise ImportError("datasets is required to get the dataset features.") from e
        if concept is None:
            concept_features = {}
        else:
            concept_features = {
                "label": concept.get_dataset_feature(),
            }
        return Features(
            {
                "gameid": Value("string"),
                "moves": Sequence(Value("string")),
                "fen": Value("string"),
                **concept_features,
            }
        )

    @staticmethod
    def concept_collate_fn(batch):
        boards = []
        labels = []
        for element in batch:
            board = LczeroBoard(element["fen"])
            for move in element["moves"]:
                board.push_san(move)
            boards.append(board)
            labels.append(element["label"])
        return boards, labels, batch

    @staticmethod
    def concept_init_grad(output, infos):
        labels = infos[0]
        rel = torch.zeros_like(output)
        for i in range(rel.shape[0]):
            rel[i, labels[i]] = output[i, labels[i]]
        return rel


@dataclass
class PuzzleData:
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
        return len(self.moves)

    @property
    def initial_board(self) -> LczeroBoard:
        board = LczeroBoard(self.fen)
        board.push(self.initial_move)
        return board

    def board_move_generator(self, all_moves: bool = False) -> Iterable[Tuple[LczeroBoard, chess.Move]]:
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
        return next(iter(self.evaluate_multiple([self], sampler, all_moves, **kwargs)))

    @staticmethod
    def compute_metrics(
        puzzles: Iterable["PuzzleData"],
        inputs: Iterable[Tuple[torch.Tensor, torch.Tensor, chess.Move]],
        all_moves: bool = False,
    ) -> Iterable[Dict[str, float]]:
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
        return self.initial_board._repr_svg_()

    @staticmethod
    def get_dataset_features():
        """Returns the features for the puzzle dataset."""
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
