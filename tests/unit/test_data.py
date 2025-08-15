"""Puzzle tests."""

import pytest
from lczerolens import LczeroBoard


from lczerolens.data import PuzzleData
from lczerolens.sampling import RandomSampler, PolicySampler


@pytest.fixture
def opening_puzzle():
    return {
        "PuzzleId": "1",
        "FEN": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "Moves": "e2e4 e7e5 d2d4 d7d5",
        "Rating": 1000,
        "RatingDeviation": 100,
        "Popularity": 1000,
        "NbPlays": 1000,
        "Themes": "Opening",
        "GameUrl": "https://lichess.org/training/1",
        "OpeningTags": "Ruy Lopez",
    }


@pytest.fixture
def easy_puzzle():
    return {
        "PuzzleId": "00008",
        "FEN": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
        "Moves": "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1",
        "Rating": 1913,
        "RatingDeviation": 75,
        "Popularity": 94,
        "NbPlays": 6230,
        "Themes": "crushing hangingPiece long middlegame",
        "GameUrl": "https://lichess.org/787zsVup/black#47",
        "OpeningTags": None,
    }


class TestPuzzle:
    def test_puzzle_creation(self, opening_puzzle):
        """Test puzzle creation."""
        puzzle = PuzzleData.from_dict(opening_puzzle)
        assert len(puzzle) == 3
        assert puzzle.rating == 1000
        assert puzzle.rating_deviation == 100
        assert puzzle.popularity == 1000
        assert puzzle.nb_plays == 1000
        assert puzzle.themes == ["Opening"]
        assert puzzle.game_url == "https://lichess.org/training/1"
        assert puzzle.opening_tags == ["Ruy", "Lopez"]

    def test_puzzle_use(self, opening_puzzle):
        """Test puzzle use."""
        puzzle = PuzzleData.from_dict(opening_puzzle)
        assert len(list(puzzle.board_move_generator())) == 2
        assert len(list(puzzle.board_move_generator(all_moves=True))) == 3


class TestRandomSampler:
    def test_puzzle_evaluation(self, opening_puzzle):
        """Test puzzle evaluation."""
        puzzle = PuzzleData.from_dict(opening_puzzle)
        sampler = RandomSampler()
        metrics = puzzle.evaluate(sampler)
        assert metrics["score"] != 1.0
        assert abs(metrics["perplexity"] - (20.0 * 30) ** 0.5) < 1e-3

    def test_puzzle_multiple_evaluation_len(self, easy_puzzle):
        """Test puzzle evaluation."""
        puzzles = [PuzzleData.from_dict(easy_puzzle) for _ in range(10)]
        sampler = RandomSampler()
        all_results = PuzzleData.evaluate_multiple(puzzles, sampler, all_moves=True, compute_metrics=False)
        assert len(list(all_results)) == 10 * 5
        results = PuzzleData.evaluate_multiple(puzzles, sampler, compute_metrics=False)
        assert len(list(results)) == 10 * 3

    def test_puzzle_multiple_evaluation(self, easy_puzzle):
        """Test puzzle evaluation."""
        puzzles = [PuzzleData.from_dict(easy_puzzle) for _ in range(10)]
        sampler = RandomSampler()
        all_results = PuzzleData.evaluate_multiple(puzzles, sampler, all_moves=True)
        assert len(list(all_results)) == 10
        results = PuzzleData.evaluate_multiple(puzzles, sampler, all_moves=False)
        assert len(list(results)) == 10

    def test_puzzle_multiple_evaluation_batch_size(self, easy_puzzle):
        """Test puzzle evaluation."""
        puzzles = [PuzzleData.from_dict(easy_puzzle) for _ in range(10)]
        sampler = RandomSampler()
        all_results = PuzzleData.evaluate_multiple(puzzles, sampler, all_moves=True, batch_size=5)
        assert len(list(all_results)) == 10
        results = PuzzleData.evaluate_multiple(puzzles, sampler, all_moves=False, batch_size=5)
        assert len(list(results)) == 10


class TestPolicySampler:
    def test_puzzle_evaluation(self, easy_puzzle, winner_model):
        """Test puzzle evaluation."""
        puzzle = PuzzleData.from_dict(easy_puzzle)
        sampler = PolicySampler(model=winner_model, use_argmax=True)
        metrics = puzzle.evaluate(sampler, all_moves=True)
        assert metrics["score"] > 0.0
        assert metrics["perplexity"] < 15.0

    def test_puzzle_multiple_evaluation(self, easy_puzzle, tiny_model):
        """Test puzzle evaluation."""
        puzzles = [PuzzleData.from_dict(easy_puzzle) for _ in range(10)]
        sampler = PolicySampler(model=tiny_model, use_argmax=False)
        all_results = PuzzleData.evaluate_multiple(puzzles, sampler, all_moves=True)
        assert len(list(all_results)) == 10
        results = PuzzleData.evaluate_multiple(puzzles, sampler, all_moves=False)
        assert len(list(results)) == 10


class TestGameData:
    def test_from_dict_without_book_exit(self):
        from lczerolens.data import GameData

        obj = {
            "gameid": "g1",
            "moves": "1. e4 e5 2. Nf3 Nc6",
        }
        gd = GameData.from_dict(obj)
        assert gd.gameid == "g1"
        assert gd.book_exit is None
        assert gd.moves == ["e4", "e5", "Nf3", "Nc6"]

    def test_from_dict_with_book_exit(self):
        from lczerolens.data import GameData

        obj = {
            "gameid": "g2",
            "moves": "1. e4 e5 { Book exit } 2. Nf3 Nc6",
        }
        gd = GameData.from_dict(obj)
        assert gd.book_exit == 2
        assert gd.moves == ["e4", "e5", "Nf3", "Nc6"]

    def test_from_dict_multiple_book_exit_raises(self):
        from lczerolens.data import GameData

        obj = {
            "gameid": "g3",
            "moves": "1. e4 e5 { Book exit } 2. Nf3 Nc6 { Book exit } 3. Bb5 a6",
        }
        with pytest.raises(ValueError):
            GameData.from_dict(obj)

    def test_from_dict_missing_keys_raise_valueerror(self):
        from lczerolens.data import GameData

        with pytest.raises(ValueError):
            GameData.from_dict({"gameid": "g4"})
        with pytest.raises(ValueError):
            GameData.from_dict({"moves": "1. e4 e5"})

    def test_to_boards_output_dict_and_skip_flags(self):
        from lczerolens.data import GameData

        gd = GameData.from_dict(
            {
                "gameid": "g5",
                "moves": "1. e4 e5 { Book exit } 2. Nf3 Nc6",
            }
        )
        # Default, includes initial position and 3 intermediate boards (skip last move)
        boards = gd.to_boards(output_dict=True)
        assert isinstance(boards, list) and len(boards) == 4
        assert all(isinstance(b, dict) for b in boards)

        # Skip book exit → no initial board, only entries after book exit index
        boards_skip_book = gd.to_boards(output_dict=True, skip_book_exit=True)
        assert len(boards_skip_book) == 1

        # Skip first n → no initial board, skip first 1 position after first move
        boards_skip_first = gd.to_boards(output_dict=True, skip_first_n=1)
        # After skipping: there should be 2 entries (for i=1 and i=2)
        assert len(boards_skip_first) == 2

    def test_to_boards_with_n_history(self):
        from lczerolens.data import GameData

        gd = GameData.from_dict({"gameid": "g6", "moves": "1. e4 e5 2. Nf3 Nc6"})

        # Test with n_history=1 (keep 1 move in stack)
        boards = gd.to_boards(n_history=1, output_dict=True)
        assert len(boards) == 4
        # Check that moves list has at most 1 move when n_history=1
        for board_dict in boards[1:]:  # Skip initial position
            assert len(board_dict["moves"]) <= 1

    def test_to_boards_with_concept(self):
        from lczerolens.data import GameData
        from lczerolens.concepts import HasPiece

        gd = GameData.from_dict({"gameid": "g7", "moves": "1. e4 e5"})
        concept = HasPiece("P")

        # Test with concept - should include labels
        boards = gd.to_boards(output_dict=True, concept=concept)
        # The new logic returns: initial board + board after e4 (but not after e5 since it's the last move)
        assert len(boards) == 2  # initial + 1 move (e4)
        assert all("label" in board for board in boards)
        # Labels should be computed values, not None
        assert all(board["label"] is not None for board in boards)

    def test_to_boards_skip_book_exit_with_concept(self):
        from lczerolens.data import GameData
        from lczerolens.concepts import HasPiece

        gd = GameData.from_dict({"gameid": "g8", "moves": "1. e4 e5 { Book exit } 2. Nf3 Nc6"})
        concept = HasPiece("P")

        # Test skip_book_exit with concept
        boards = gd.to_boards(output_dict=True, skip_book_exit=True, concept=concept)
        assert len(boards) == 1  # Only after book exit
        assert "label" in boards[0]
        assert boards[0]["label"] is not None

    def test_to_boards_return_boards(self):
        from lczerolens.data import GameData
        from lczerolens.board import LczeroBoard

        gd = GameData.from_dict({"gameid": "g6", "moves": "1. e4 e5 2. Nf3 Nc6"})

        # Test default (n_history=0)
        boards = gd.to_boards(output_dict=False)
        assert isinstance(boards, list) and len(boards) == 4
        assert all(isinstance(b, LczeroBoard) for b in boards)

        # Test with n_history=2
        boards_with_history = gd.to_boards(output_dict=False, n_history=2)
        assert isinstance(boards_with_history, list) and len(boards_with_history) == 4
        assert all(isinstance(b, LczeroBoard) for b in boards_with_history)
        # Check that boards have move stack with at most 2 moves
        for board in boards_with_history[1:]:  # Skip initial position
            assert len(board.move_stack) <= 2

    def test_board_collate_fn(self):
        from lczerolens.data import BoardData
        from lczerolens.board import LczeroBoard

        # Use SAN moves, as expected by collate
        batch = [
            {
                "fen": LczeroBoard().fen(),
                "moves": ["e4", "e5", "Nf3"],
            }
        ]
        boards = BoardData.board_collate_fn(batch)
        assert isinstance(boards, list) and len(boards) == 1
        assert isinstance(boards[0], LczeroBoard)
        assert len(list(boards[0].move_stack)) == 3

    def test_board_collate_fn_with_concept(self):
        from lczerolens.data import BoardData
        from lczerolens.board import LczeroBoard

        # Test BoardData.board_collate_fn (concept parameter not supported)
        batch = [
            {
                "fen": LczeroBoard().fen(),
                "moves": ["e4", "e5"],
            }
        ]
        boards = BoardData.board_collate_fn(batch)
        assert isinstance(boards, list) and len(boards) == 1
        assert isinstance(boards[0], LczeroBoard)
        assert len(list(boards[0].move_stack)) == 2

    def test_get_dataset_features(self):
        datasets = pytest.importorskip("datasets")
        from lczerolens.data import GameData

        features = GameData.get_dataset_features()
        data = [{"gameid": "g7", "moves": ["e4", "e5"]}]
        ds = datasets.Dataset.from_list(data, features=features)
        assert set(ds.features.keys()) == {"gameid", "moves"}
        assert ds.features["gameid"].dtype == "string"
        assert ds.features["moves"].feature.dtype == "string"


class TestBoardData:
    def test_board_data_creation_with_label(self):
        from lczerolens.data import BoardData

        # Test creation with label
        board_data = BoardData(
            gameid="test_game",
            moves=["e4", "e5"],
            fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            label=1,
        )
        assert board_data.gameid == "test_game"
        assert board_data.moves == ["e4", "e5"]
        assert board_data.fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        assert board_data.label == 1

    def test_board_data_creation_without_label(self):
        from lczerolens.data import BoardData

        # Test creation without label
        board_data = BoardData(
            gameid="test_game", moves=["e4", "e5"], fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        )
        assert board_data.label is None

    def test_get_dataset_features_with_and_without_concept(self):
        datasets = pytest.importorskip("datasets")
        from lczerolens.data import BoardData
        from lczerolens.concepts import HasPiece

        # Without concept - still need to provide label field (will be null)
        features = BoardData.get_dataset_features()
        data = [{"gameid": "g8", "moves": ["e4"], "fen": LczeroBoard().fen(), "label": None}]
        ds = datasets.Dataset.from_list(data, features=features)
        assert set(ds.features.keys()) == {"gameid", "moves", "fen", "label"}

        # With a binary concept → label feature added
        concept = HasPiece("P")
        features_with_label = BoardData.get_dataset_features(concept)
        data_l = [{"gameid": "g9", "moves": ["e4"], "fen": LczeroBoard().fen(), "label": 1}]
        ds_l = datasets.Dataset.from_list(data_l, features=features_with_label)
        assert "label" in ds_l.features

    def test_concept_collate_fn(self):
        from lczerolens.data import BoardData

        batch = [
            {
                "fen": LczeroBoard().fen(),
                "moves": ["e4", "e5"],
                "label": 1,
            }
        ]
        boards, labels, infos = BoardData.concept_collate_fn(batch)
        assert len(boards) == 1 and isinstance(boards[0], LczeroBoard)
        assert labels == [1]
        assert infos == batch

    def test_concept_init_grad(self):
        import torch
        from lczerolens.data import BoardData

        output = torch.randn(3, 5)
        labels = [0, 2, 4]
        rel = BoardData.concept_init_grad(output, (labels,))
        assert rel.shape == output.shape
        for i, j in enumerate(labels):
            assert rel[i, j] == output[i, j]
            # Off-diagonal should be zero at least for a few
            assert torch.all(rel[i, torch.arange(output.shape[1]) != j] == 0)


class TestPuzzleDataFeatures:
    def test_repr_svg_smoke(self, easy_puzzle):
        # Ensure _repr_svg_ produces an SVG string
        puzzle = PuzzleData.from_dict(easy_puzzle)
        svg = puzzle._repr_svg_()
        assert isinstance(svg, str) and "<svg" in svg

    def test_get_dataset_features(self):
        datasets = pytest.importorskip("datasets")
        features = PuzzleData.get_dataset_features()
        sample = {
            "PuzzleId": "pid",
            "FEN": "8/8/8/8/8/8/8/8 w - - 0 1",
            "Moves": "a2a3 a7a6",
            "Rating": 1000,
            "RatingDeviation": 100,
            "Popularity": 10,
            "NbPlays": 20,
            "Themes": "",
            "GameUrl": "https://example.org",
            "OpeningTags": "",
        }
        ds = datasets.Dataset.from_list([sample], features=features)
        assert set(ds.features.keys()) == {
            "PuzzleId",
            "FEN",
            "Moves",
            "Rating",
            "RatingDeviation",
            "Popularity",
            "NbPlays",
            "Themes",
            "GameUrl",
            "OpeningTags",
        }
