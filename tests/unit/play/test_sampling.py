"""Sampling tests."""

from lczerolens.play.sampling import ModelSampler, SelfPlay, PolicySampler, RandomSampler
from lczerolens.board import LczeroBoard


class TestRandomSampler:
    def test_get_utilities(self):
        """Test get_utilities method."""
        board = LczeroBoard()
        sampler = RandomSampler()
        utility, _, _ = next(iter(sampler.get_utilities([board, board])))
        assert utility.shape[0] == 20


class TestModelSampler:
    def test_get_utilities_tiny(self, tiny_model):
        """Test get_utilities method."""
        board = LczeroBoard()
        sampler = ModelSampler(tiny_model, use_argmax=False)
        utility, _, _ = next(iter(sampler.get_utilities([board, board])))
        assert utility.shape[0] == 20

    def test_get_utilities_winner(self, winner_model):
        """Test get_utilities method."""
        board = LczeroBoard()
        sampler = ModelSampler(winner_model, use_argmax=False)
        utility, _, _ = next(iter(sampler.get_utilities([board, board])))
        assert utility.shape[0] == 20

    def test_policy_sampler_tiny(self, tiny_model):
        """Test policy_sampler method."""
        board = LczeroBoard()
        sampler = PolicySampler(tiny_model, use_argmax=False)
        utility, _, _ = next(iter(sampler.get_utilities([board, board])))
        assert utility.shape[0] == 20


class TestSelfPlay:
    def test_play(self, tiny_model, winner_model):
        """Test play method."""
        board = LczeroBoard()
        white = ModelSampler(tiny_model, use_argmax=False)
        black = ModelSampler(winner_model, use_argmax=False)
        self_play = SelfPlay(white=white, black=black)
        logs = []

        def report_fn(log, to_play):
            logs.append((log, to_play))

        game, board = self_play.play(board=board, max_moves=10, report_fn=report_fn)

        assert len(game) == len(logs) == 10


class TestBatchedPolicySampler:
    def test_batched_policy_sampler_ag(self, tiny_model):
        """Test batched_policy_sampler method."""
        boards = [LczeroBoard() for _ in range(10)]

        sampler_ag = PolicySampler(tiny_model, use_argmax=True)
        moves = sampler_ag.get_next_moves(boards)
        assert len(list(moves)) == 10
        assert all([move == moves[0] for move in moves])

    def test_batched_policy_sampler_no_ag(self, tiny_model):
        """Test batched_policy_sampler method."""
        boards = [LczeroBoard() for _ in range(10)]

        sampler_no_ag = PolicySampler(tiny_model, use_argmax=False)
        moves = sampler_no_ag.get_next_moves(boards)
        assert len(list(moves)) == 10

    def test_batched_policy_sampler_no_ag_sub(self, tiny_model):
        """Test batched_policy_sampler method."""
        boards = [LczeroBoard() for _ in range(10)]

        sampler_no_ag = PolicySampler(tiny_model, use_argmax=False, use_suboptimal=True)
        moves = sampler_no_ag.get_next_moves(boards)
        assert len(list(moves)) == 10
