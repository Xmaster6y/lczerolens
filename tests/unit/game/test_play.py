"""Wrapper tests."""

import chess

from lczerolens.game import WrapperSampler, SelfPlay, PolicySampler, BatchedPolicySampler


class TestWrapperSampler:
    def test_get_utility_tiny(self, tiny_wrapper):
        """Test get_utility method."""
        board = chess.Board()
        sampler = WrapperSampler(wrapper=tiny_wrapper)
        utility, _, _ = sampler.get_utility(board)
        assert utility.shape[0] == 20

    def test_get_utility_winner(self, winner_wrapper):
        """Test get_utility method."""
        board = chess.Board()
        sampler = WrapperSampler(wrapper=winner_wrapper)
        utility, _, _ = sampler.get_utility(board)
        assert utility.shape[0] == 20

    def test_policy_sampler_tiny(self, tiny_wrapper):
        """Test policy_sampler method."""
        board = chess.Board()
        sampler = PolicySampler(wrapper=tiny_wrapper)
        utility, _, _ = sampler.get_utility(board)
        assert utility.shape[0] == 20


class TestSelfPlay:
    def test_play(self, tiny_wrapper, winner_wrapper):
        """Test play method."""
        board = chess.Board()
        white = WrapperSampler(wrapper=tiny_wrapper)
        black = WrapperSampler(wrapper=winner_wrapper)
        self_play = SelfPlay(white=white, black=black)
        logs = []

        def report_fn(log, to_play):
            logs.append((log, to_play))

        game, board = self_play.play(board=board, max_moves=10, report_fn=report_fn)

        assert len(game) == len(logs) == 10


class TestBatchedPolicySampler:
    def test_batched_policy_sampler_ag(self, tiny_wrapper):
        """Test batched_policy_sampler method."""
        boards = [chess.Board() for _ in range(10)]

        sampler_ag = BatchedPolicySampler(wrapper=tiny_wrapper, use_argmax=True)
        moves = sampler_ag.get_next_moves(boards)
        assert len(list(moves)) == 10
        assert all([move == moves[0] for move in moves])

    def test_batched_policy_sampler_no_ag(self, tiny_wrapper):
        """Test batched_policy_sampler method."""
        boards = [chess.Board() for _ in range(10)]

        sampler_no_ag = BatchedPolicySampler(wrapper=tiny_wrapper, use_argmax=False)
        moves = sampler_no_ag.get_next_moves(boards)
        assert len(list(moves)) == 10
