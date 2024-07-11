"""Preproces functions for chess puzzles."""

from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Optional, Iterable

import chess
import torch
from datasets import Features, Value
from itertools import tee, chain

from lczerolens.encodings import move as move_encodings
from .sampling import Sampler


PUZZLE_DATASET_FEATURES = Features(
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


@dataclass
class Puzzle:
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
    def from_dict(cls, obj: Dict[str, Union[str, int, None]]) -> "Puzzle":
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
    def initial_board(self) -> chess.Board:
        board = chess.Board(self.fen)
        board.push(self.initial_move)
        return board

    def board_move_generator(self, all_moves: bool = False) -> Iterable[chess.Board]:
        board = self.initial_board
        initial_turn = board.turn
        for move in self.moves:
            if not all_moves and board.turn != initial_turn:
                continue
            yield board.copy(), move
            board.push(move)

    @classmethod
    def evaluate_multiple(
        cls,
        puzzles: Iterable["Puzzle"],
        sampler: Sampler,
        use_perplexity: bool = False,
        all_moves: bool = False,
        **kwargs,
    ) -> Iterable[Tuple[float, Optional[float]]]:
        metric_puzzles, board_move_puzzles = tee(puzzles)
        board_move_generator = chain.from_iterable(
            puzzle.board_move_generator(all_moves) for puzzle in board_move_puzzles
        )

        def board_generator():
            for board, _ in board_move_generator:
                yield board

        util_boards, move_boards = tee(board_generator())

        def metric_inputs_generator():
            util_gen = sampler.get_utility(util_boards, **kwargs)
            for board, (utility, legal_indices, _) in zip(move_boards, util_gen):
                predicted_move = sampler.choose_move(board, utility, legal_indices)
                yield utility, legal_indices, predicted_move

        return cls.compute_metrics(metric_puzzles, metric_inputs_generator(), use_perplexity)

    def evaluate(
        self, sampler: Sampler, use_perplexity: bool = False, all_moves: bool = False, **kwargs
    ) -> Tuple[float, Optional[float]]:
        return next(iter(self.evaluate_multiple([self], sampler, use_perplexity, all_moves, **kwargs)))

    @staticmethod
    def compute_metrics(
        puzzles: Iterable["Puzzle"],
        inputs: Iterable[Tuple[torch.Tensor, torch.Tensor, chess.Move]],
        use_perplexity: bool = False,
    ) -> Iterable[Tuple[float, Optional[float]]]:
        iter_inputs = iter(inputs)
        for puzzle in puzzles:
            total = 0
            score = 0.0
            perplexity = 1.0 if use_perplexity else 0.0
            for board, move in puzzle.board_move_generator():
                utility, legal_indices, predicted_move = next(iter_inputs)
                if use_perplexity:
                    index = move_encodings.encode_move(move, board.turn)
                    probs = torch.softmax(utility, dim=0)
                    perplexity *= probs[legal_indices == index].item()
                if predicted_move == move:
                    score += 1
                total += 1
            score /= total
            if not use_perplexity or perplexity == 0:
                perplexity = None
            else:
                perplexity = perplexity ** (-1 / total)
            yield score, perplexity

    def __repr__(self) -> str:
        return self.initial_board.__repr__()

    def _repr_svg_(self) -> str:
        return self.initial_board._repr_svg_()
