"""Classes for playing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict, Iterable

import chess
import torch
from torch.distributions import Categorical
from itertools import tee

from lczerolens.model import LczeroModel
from lczerolens.board import LczeroBoard


class Sampler(ABC):
    @abstractmethod
    def get_utilities(
        self, boards: Iterable[LczeroBoard], **kwargs
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        """Get the utility of the board.

        Parameters
        ----------
        boards : Iterable[LczeroBoard]
            The boards to evaluate.

        Returns
        -------
        Iterable[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]
            The iterable over utilities, legal indices, and log dictionaries.
        """
        pass

    def choose_move(self, board: LczeroBoard, utility: torch.Tensor, legal_indices: torch.Tensor) -> chess.Move:
        """Choose the next moves.

        Parameters
        ----------
        board : LczeroBoard
            The board.
        utility : torch.Tensor
            The utility of the board.
        legal_indices : torch.Tensor
            The legal indices.

        Returns
        -------
        Iterable[chess.Move]
            The iterable over the moves.
        """
        m = Categorical(logits=utility)
        idx = m.sample()
        return board.decode_move(legal_indices[idx])

    def get_next_moves(self, boards: Iterable[LczeroBoard], **kwargs) -> Iterable[Tuple[chess.Move, Dict[str, float]]]:
        """Get the next move.

        Parameters
        ----------
        boards : Iterable[LczeroBoard]
            The boards to evaluate.

        Returns
        -------
        Iterable[Tuple[chess.Move, Dict[str, float]]]
            The iterable over the moves and log dictionaries.
        """
        util_boards, move_boards = tee(boards)
        for board, (utility, legal_indices, to_log) in zip(move_boards, self.get_utilities(util_boards, **kwargs)):
            predicted_move = self.choose_move(board, utility, legal_indices)
            yield predicted_move, to_log


class RandomSampler(Sampler):
    def get_utilities(
        self, boards: Iterable[LczeroBoard], **kwargs
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        for board in boards:
            legal_indices = board.get_legal_indices()
            utilities = torch.ones_like(legal_indices, dtype=torch.float32)
            yield utilities, legal_indices, {}


@dataclass
class ModelSampler(Sampler):
    model: LczeroModel
    use_argmax: bool = True
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    draw_score: float = 0.0
    m_max: float = 0.0345
    m_slope: float = 0.0027
    k_0: float = 0.0
    k_1: float = 1.6521
    k_2: float = -0.6521
    q_threshold: float = 0.8

    def choose_move(self, board: LczeroBoard, utility: torch.Tensor, legal_indices: torch.Tensor) -> chess.Move:
        if self.use_argmax:
            idx = utility.argmax()
            return board.decode_move(legal_indices[idx])
        return super().choose_move(board, utility, legal_indices)

    @torch.no_grad
    def get_utilities(
        self, boards: Iterable[LczeroBoard], **kwargs
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        batch_size = kwargs.pop("batch_size", -1)

        for legal_indices, batch_stats in self._get_batched_stats(boards, batch_size, **kwargs):
            to_log = {}
            utility = 0
            q_values = self._get_q_values(batch_stats, to_log)
            utility += self.alpha * q_values
            utility += self.beta * self._get_m_values(batch_stats, q_values, to_log)
            utility += self.gamma * self._get_p_values(batch_stats, legal_indices, to_log)
            to_log["max_utility"] = utility.max().item()
            yield utility, legal_indices, to_log

    def _get_batched_stats(self, boards, batch_size, use_next_boards=True, **kwargs):
        next_batch = []
        next_legal_indices = []

        def generator(next_batch, next_legal_indices):
            all_stats = self.model(*next_batch, **kwargs)
            offset = 0
            for legal_indices in next_legal_indices:
                n_boards = legal_indices.shape[0] + 1 if use_next_boards else 1
                batch_stats = all_stats[offset : offset + n_boards]
                offset += n_boards
                yield legal_indices, batch_stats

        for board in boards:
            legal_indices = board.get_legal_indices()
            next_boards = list(board.get_next_legal_boards()) if use_next_boards else []
            if len(next_batch) + len(next_boards) + 1 > batch_size and batch_size != -1:
                yield from generator(next_batch, next_legal_indices)
                next_batch = []
                next_legal_indices = []
            next_batch.extend([board] + next_boards)
            next_legal_indices.append(legal_indices)
        if next_batch:
            yield from generator(next_batch, next_legal_indices)

    def _get_q_values(self, batch_stats, to_log):
        if "value" in batch_stats.keys():
            to_log["value"] = batch_stats["value"][0].item()
            return batch_stats["value"][1:, 0]
        elif "wdl" in batch_stats.keys():
            to_log["wdl_w"] = batch_stats["wdl"][0][0].item()
            to_log["wdl_d"] = batch_stats["wdl"][0][1].item()
            to_log["wdl_l"] = batch_stats["wdl"][0][2].item()
            scores = torch.tensor([1, self.draw_score, -1])
            return batch_stats["wdl"][1:] @ scores
        return torch.zeros(batch_stats.batch_size[0] - 1)

    def _get_m_values(self, batch_stats, q_values, to_log):
        if "mlh" in batch_stats.keys():
            to_log["mlh"] = batch_stats["mlh"][0].item()
            delta_m_values = self.m_slope * (batch_stats["mlh"][1:, 0] - batch_stats["mlh"][0, 0])
            delta_m_values.clamp_(-self.m_max, self.m_max)
            scaled_q_values = torch.relu(q_values.abs() - self.q_threshold) / (1 - self.q_threshold)
            poly_q_values = self.k_0 + self.k_1 * scaled_q_values + self.k_2 * scaled_q_values**2
            return -q_values.sign() * delta_m_values * poly_q_values
        return torch.zeros(batch_stats.batch_size[0] - 1)

    def _get_p_values(
        self,
        batch_stats,
        legal_indices,
        to_log,
    ):
        if "policy" in batch_stats.keys():
            legal_policy = batch_stats["policy"][0].gather(0, legal_indices)
            to_log["max_legal_policy"] = legal_policy.max().item()
            return legal_policy
        return torch.zeros_like(legal_indices)


@dataclass
class PolicySampler(ModelSampler):
    use_suboptimal: bool = False

    @torch.no_grad
    def get_utilities(
        self, boards: Iterable[LczeroBoard], **kwargs
    ) -> Iterable[Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]]:
        batch_size = kwargs.pop("batch_size", -1)

        to_log = {}
        for legal_indices, batch_stats in self._get_batched_stats(boards, batch_size, use_next_boards=False, **kwargs):
            legal_policy = batch_stats["policy"][0].gather(0, legal_indices.to(batch_stats["policy"].device))
            if self.use_suboptimal:
                idx = legal_policy.argmax()
                legal_policy[idx] = torch.tensor(-1e3)
            yield legal_policy, legal_indices, to_log


@dataclass
class SelfPlay:
    """A class for generating games."""

    white: Sampler
    black: Sampler

    def play(
        self,
        board: Optional[LczeroBoard] = None,
        max_moves: int = 100,
        to_play: chess.Color = chess.WHITE,
        report_fn: Optional[Callable[[dict, chess.Color], None]] = None,
        white_kwargs: Optional[Dict] = None,
        black_kwargs: Optional[Dict] = None,
    ):
        """
        Plays a game.
        """
        if board is None:
            board = LczeroBoard()
        if white_kwargs is None:
            white_kwargs = {}
        if black_kwargs is None:
            black_kwargs = {}
        game = []
        if to_play == chess.BLACK:
            move, _ = next(iter(self.black.get_next_moves([board], **black_kwargs)))
            board.push(move)
            game.append(move)
        for _ in range(max_moves):
            if board.is_game_over() or len(game) >= max_moves:
                break
            move, to_log = next(iter(self.white.get_next_moves([board], **white_kwargs)))
            if report_fn is not None:
                report_fn(to_log, board.turn)
            board.push(move)
            game.append(move)

            if board.is_game_over() or len(game) >= max_moves:
                break
            move, to_log = next(iter(self.black.get_next_moves([board], **black_kwargs)))
            if report_fn is not None:
                report_fn(to_log, board.turn)
            board.push(move)
            game.append(move)
            if board.is_game_over() or len(game) >= max_moves:
                break
        return game, board
