"""Preproces functions for chess puzzles."""

from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Optional, Iterable

import chess
import torch
from datasets import Features, Value
from itertools import tee, chain

from lczerolens.board import LczeroBoard
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
        puzzles: Iterable["Puzzle"],
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
        puzzles: Iterable["Puzzle"],
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
