"""Tests for the unified natural search contract."""

from dataclasses import replace
from types import SimpleNamespace

import chess
import chess.variant
import pytest
import torch
from torch import nn

from lczerolens import (
    Depth,
    LczeroModel,
    LczeroSearch,
    Nodes,
    ReferenceSearch,
    Simulations,
    Time,
    Visits,
)
from lczerolens._codec import encode_move
from lczerolens.evaluator import LczeroEvaluator
from lczerolens.provenance import ChessPlayer
from lczerolens.search import lczero as lczero_module
from lczerolens.search.result import SearchEvidenceUnavailable, SearchResult, SearchRoot
from lczerolens.search.trace import (
    PositionEvaluation,
    PrincipalVariation,
    RootSnapshot,
    SearchCapability,
    SearchProvenance,
    SearchTrace,
    ValuePerspective,
)


class FixtureNetwork(nn.Module):
    def forward(self, planes):
        batch = planes.shape[0]
        policy = torch.zeros((batch, 1858), device=planes.device)
        board = chess.Board()
        policy[:, encode_move(board, chess.Move.from_uci("e2e4"))] = 2.0
        return policy, torch.full((batch,), 0.25, device=planes.device)


class PolicyOnlyNetwork(nn.Module):
    def forward(self, planes):
        return torch.zeros((planes.shape[0], 1858), device=planes.device)


def fixture_evaluator():
    return LczeroEvaluator(LczeroModel(FixtureNetwork(), out_keys=["policy", "value"]))


@pytest.mark.parametrize("limit", (Nodes(1), Visits(1), Simulations(1), Depth(1)))
def test_count_limits_are_typed_positive_units(limit):
    assert limit.value == 1
    assert limit.unit.value == type(limit).__name__.lower()


@pytest.mark.parametrize("limit_type", (Nodes, Visits, Simulations, Depth))
@pytest.mark.parametrize("value", (0, -1, 1.5, True))
def test_count_limits_reject_non_positive_integers(limit_type, value):
    with pytest.raises(ValueError, match="positive integer"):
        limit_type(value)


def test_time_limit_accepts_fractional_milliseconds_and_rejects_invalid_values():
    assert Time(1.5).value == 1.5
    assert Time(1.5).unit.value == "time_ms"
    for value in (0, -1, float("inf"), True, "1"):
        with pytest.raises(ValueError, match="finite, positive"):
            Time(value)


def test_reference_search_returns_replayable_natural_result_from_plain_board():
    board = chess.Board()
    search = ReferenceSearch(fixture_evaluator(), c_puct=1.0)

    result = search.run(board, Simulations(2))
    repeat = search.run(board, Simulations(2))

    assert result == repeat
    assert result.move in board.legal_moves
    assert result.evaluation is not None
    assert result.root[result.move].visits is not None
    assert result.trace.snapshots[-1].budget.requested == 2
    assert result.trace.snapshots[-1].budget.observed == 2
    assert result.has_root_actions
    assert result.has_snapshots
    assert result.has_events
    assert result.is_replayable
    assert result.capability is SearchCapability.REPLAYABLE
    assert result.trace.has_root_actions
    assert result.trace.has_snapshots
    assert result.trace.has_events
    assert result.trace.is_replayable


def test_reference_search_preserves_retained_history_and_rejects_unsupported_inputs():
    board = chess.Board()
    board.push_uci("g1f3")
    result = ReferenceSearch(fixture_evaluator()).run(board, Simulations(1))

    assert result.trace.root_start_fen == chess.STARTING_FEN
    assert result.trace.root_move_history == ("g1f3",)
    with pytest.raises(ValueError, match="only Simulations"):
        ReferenceSearch(fixture_evaluator()).run(chess.Board(), Nodes(1))
    with pytest.raises(TypeError, match="python-chess Board"):
        ReferenceSearch(fixture_evaluator()).run(object(), Simulations(1))
    with pytest.raises(ValueError, match="non-terminal"):
        ReferenceSearch(fixture_evaluator()).run(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"), Simulations(1))
    with pytest.raises(ValueError, match="standard chess"):
        ReferenceSearch(fixture_evaluator()).run(chess.variant.AtomicBoard(), Simulations(1))
    with pytest.raises(ValueError, match="standard chess"):
        ReferenceSearch(fixture_evaluator()).run(chess.Board(chess960=True), Simulations(1))
    with pytest.raises(TypeError, match="LczeroEvaluator"):
        ReferenceSearch(object())


def test_reference_search_requires_a_scalar_evaluator_value():
    evaluator = LczeroEvaluator(LczeroModel(PolicyOnlyNetwork(), out_keys=["policy"]))

    with pytest.raises(ValueError, match="scalar value"):
        ReferenceSearch(evaluator).run(chess.Board(), Simulations(1))


def test_lczero_search_maps_supported_limits_and_preserves_root_only_absence(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="info nodes 2\nbestmove e2e4\n", stderr="")

    monkeypatch.setattr(lczero_module, "_run_uci_process", fake_run)
    search = LczeroSearch(executable="lc0", network="weights.pb.gz", engine_version="v-test", timeout=3)

    result = search.run(chess.Board(), Nodes(2))

    assert result.move == chess.Move.from_uci("e2e4")
    assert not result.has_root_actions
    assert result.has_snapshots
    assert not result.has_events
    assert not result.is_replayable
    with pytest.raises(SearchEvidenceUnavailable, match="root action statistics"):
        result.root
    command = calls[0][0][1]
    assert "setoption name WeightsFile value weights.pb.gz" in command
    assert "setoption name VerboseMoveStats value true" in command
    assert "go nodes 2" in command


def test_lczero_search_supports_whole_time_and_rejects_unsupported_limits(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="info time 7\nbestmove e2e4\n", stderr="")

    monkeypatch.setattr(lczero_module, "_run_uci_process", fake_run)
    search = LczeroSearch(executable="lc0", network="weights", engine_version="v-test")

    result = search.run(chess.Board(), Time(10))
    assert result.trace.snapshots[0].budget.requested == 10
    assert result.trace.snapshots[0].budget.observed == 7
    with pytest.raises(ValueError, match="whole-millisecond"):
        search.run(chess.Board(), Time(1.5))
    with pytest.raises(ValueError, match="only Nodes and Time"):
        search.run(chess.Board(), Visits(1))
    with pytest.raises(TypeError, match="python-chess Board"):
        search.run(object(), Nodes(1))
    with pytest.raises(ValueError, match="standard chess"):
        search.run(chess.variant.AtomicBoard(), Nodes(1))
    with pytest.raises(ValueError, match="standard chess"):
        search.run(chess.Board(chess960=True), Nodes(1))


def test_lczero_search_constructor_rejects_ambiguous_runtime_configuration():
    with pytest.raises(ValueError, match="network"):
        LczeroSearch(executable="lc0", network="", engine_version="v-test")
    with pytest.raises(ValueError, match="engine version"):
        LczeroSearch(executable="lc0", network="weights", engine_version="")
    with pytest.raises(ValueError, match="timeout"):
        LczeroSearch(executable="lc0", network="weights", engine_version="v-test", timeout=0)


def test_search_result_and_root_reject_inconsistent_or_unavailable_evidence():
    result = ReferenceSearch(fixture_evaluator()).run(chess.Board(), Simulations(1))
    assert result.root == SearchRoot(tuple(result.root.values()))
    assert result.root != object()
    with pytest.raises(ValueError, match="canonical UCI"):
        SearchRoot(tuple(reversed(tuple(result.root.values()))))
    with pytest.raises(ValueError, match="unique"):
        SearchRoot((result.root[result.move], result.root[result.move]))
    with pytest.raises(ValueError, match="legal"):
        replace(result, move=chess.Move.from_uci("e2e5"))
    with pytest.raises(ValueError, match="final trace selection"):
        replace(result, move=chess.Move.from_uci("d2d4"))
    with pytest.raises(ValueError, match="contain the selected"):
        replace(
            result,
            _root=SearchRoot(tuple(action for move, action in result.root.items() if move != result.move.uci())),
        )
    with pytest.raises(ValueError, match="principal variation"):
        replace(result, principal_variation=PrincipalVariation(("d2d4",)))
    with pytest.raises(ValueError, match="match the selected root action"):
        replace(result, principal_variation=PrincipalVariation((result.move.uci(),)))
    with pytest.raises(TypeError, match="SearchTrace"):
        SearchResult.from_trace(object())

    root = chess.Board()
    no_selection = SearchTrace(
        root_fen=root.fen(),
        root_player=ChessPlayer.WHITE,
        capability=SearchCapability.ROOT_RESULT,
        provenance=SearchProvenance("fixture"),
        snapshots=(RootSnapshot(0, evaluation=PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=0.0)),),
    )
    with pytest.raises(SearchEvidenceUnavailable, match="selected move"):
        SearchResult.from_trace(no_selection)
