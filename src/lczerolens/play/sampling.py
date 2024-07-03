"""Classes for playing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable, List, Generator

import chess
import torch
from torch.distributions import Categorical

from lczerolens.encodings import move as move_encodings
from lczerolens.model import LczeroModel


class Sampler(ABC):
    @abstractmethod
    def get_next_move(self, board: chess.Board):
        pass


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

    @torch.no_grad
    def get_utility(
        self,
        board: chess.Board,
    ):
        to_log = {}
        next_legal_boards = move_encodings.get_next_legal_boards(board)
        legal_indices = move_encodings.get_legal_indices(board)
        all_stats = self.model(board, *next_legal_boards)
        utility = 0
        q_values = self._get_q_values(all_stats, to_log)
        utility += self.alpha * q_values
        utility += self.beta * self._get_m_values(all_stats, q_values, to_log)
        utility += self.gamma * self._get_p_values(all_stats, legal_indices, to_log)
        to_log["max_utility"] = utility.max().item()
        return utility, legal_indices, to_log

    def _get_q_values(self, all_stats, to_log):
        if "value" in all_stats.keys():
            to_log["value"] = all_stats["value"][0].item()
            return all_stats["value"][1:, 0]
        elif "wdl" in all_stats.keys():
            to_log["wdl_w"] = all_stats["wdl"][0][0].item()
            to_log["wdl_d"] = all_stats["wdl"][0][1].item()
            to_log["wdl_l"] = all_stats["wdl"][0][2].item()
            scores = torch.tensor([1, self.draw_score, -1])
            return all_stats["wdl"][1:] @ scores
        else:
            return torch.zeros(all_stats.batch_size[0] - 1)

    def _get_m_values(self, all_stats, q_values, to_log):
        if "mlh" in all_stats.keys():
            to_log["mlh"] = all_stats["mlh"][0].item()
            delta_m_values = self.m_slope * (all_stats["mlh"][1:, 0] - all_stats["mlh"][0, 0])
            delta_m_values.clamp_(-self.m_max, self.m_max)
            scaled_q_values = torch.relu(q_values.abs() - self.q_threshold) / (1 - self.q_threshold)
            poly_q_values = self.k_0 + self.k_1 * scaled_q_values + self.k_2 * scaled_q_values**2
            return -q_values.sign() * delta_m_values * poly_q_values
        else:
            return torch.zeros(all_stats.batch_size[0] - 1)

    def _get_p_values(
        self,
        all_stats,
        legal_indices,
        to_log,
    ):
        if "policy" in all_stats.keys():
            legal_policy = all_stats["policy"][0].gather(0, legal_indices)
            to_log["max_legal_policy"] = legal_policy.max().item()
            return legal_policy
        else:
            return torch.zeros_like(legal_indices)

    def get_next_move(self, board: chess.Board):
        utility, legal_indices, to_log = self.get_utility(board)
        if self.use_argmax:
            idx = utility.argmax()
        else:
            m = Categorical(logits=utility)
            idx = m.sample()
        move = move_encodings.decode_move(legal_indices[idx], board)
        return move, to_log


class PolicySampler(ModelSampler):
    @torch.no_grad
    def get_utility(
        self,
        board: chess.Board,
    ):
        to_log = {}
        legal_indices = move_encodings.get_legal_indices(board)
        all_stats = self.model(board)
        utility = self._get_p_values(all_stats, legal_indices, to_log)
        to_log["max_utility"] = utility.max().item()
        if "value" in all_stats.keys():
            to_log["value"] = all_stats["value"][0].item()
        elif "wdl" in all_stats.keys():
            to_log["wdl_w"] = all_stats["wdl"][0][0].item()
            to_log["wdl_d"] = all_stats["wdl"][0][1].item()
            to_log["wdl_l"] = all_stats["wdl"][0][2].item()
        return utility, legal_indices, to_log


@dataclass
class SelfPlay:
    """A class for generating games."""

    white: Sampler
    black: Sampler

    def play(
        self,
        board: Optional[chess.Board] = None,
        max_moves: int = 100,
        to_play: chess.Color = chess.WHITE,
        report_fn: Optional[Callable[[dict, chess.Color], None]] = None,
    ):
        """
        Plays a game.
        """
        if board is None:
            board = chess.Board()
        game = []
        if to_play == chess.BLACK:
            move, _ = self.black.get_next_move(board)
            board.push(move)
            game.append(move)
        for _ in range(max_moves):
            if board.is_game_over() or len(game) >= max_moves:
                break
            move, to_log = self.white.get_next_move(board)
            if report_fn is not None:
                report_fn(to_log, board.turn)
            board.push(move)
            game.append(move)

            if board.is_game_over() or len(game) >= max_moves:
                break
            move, to_log = self.black.get_next_move(board)
            if report_fn is not None:
                report_fn(to_log, board.turn)
            board.push(move)
            game.append(move)
            if board.is_game_over() or len(game) >= max_moves:
                break
        return game, board


@dataclass
class BatchedPolicySampler:
    model: LczeroModel
    use_argmax: bool = True
    use_suboptimal: bool = False

    @torch.no_grad
    def get_next_moves(
        self,
        boards: List[chess.Board],
    ) -> Generator[chess.Move, None, None]:
        all_stats = self.model(*boards)
        for board, policy in zip(boards, all_stats["policy"]):
            indices = move_encodings.get_legal_indices(board)
            legal_policy = policy.gather(0, indices)
            if self.use_argmax:
                idx = legal_policy.argmax()
            else:
                if self.use_suboptimal:
                    idx = legal_policy.argmax()
                    legal_policy[idx] = torch.tensor(-1e3)
                m = Categorical(logits=legal_policy)
                idx = m.sample()
            yield list(board.legal_moves)[idx]
