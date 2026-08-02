import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lczerolens import lc0_adapter
from lczerolens.lc0_adapter import Lc0OutputError, Lc0ProcessAdapter, Lc0RootSnapshotParser, Lc0SearchRequest
from lczerolens.search_trace import SearchCapability, SearchCapabilityError


ROOT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_parses_versioned_lc0_root_snapshot_fixture():
    fixture = Path(__file__).parents[1] / "assets" / "lc0_root_snapshot_v1.txt"
    trace = Lc0RootSnapshotParser().parse(
        fixture.read_text().splitlines(),
        request=Lc0SearchRequest(ROOT_FEN, nodes=256, options={"VerboseMoveStats": True}),
        engine_version="v0.31.2",
        network="test-network",
        network_checksum="sha256:test",
    )
    snapshot = trace.snapshots[0]
    e4, d4 = snapshot.actions or ()
    assert trace.capability is SearchCapability.ROOT_SNAPSHOTS
    assert snapshot.selection and snapshot.selection.move == "e2e4"
    assert snapshot.budget and snapshot.budget.requested == snapshot.budget.observed == 256
    assert (e4.statistics.prior, e4.statistics.visits, e4.statistics.mean_value, e4.statistics.exploration) == (
        0.6,
        192,
        0.25,
        0.125,
    )
    assert e4.statistics.total_value == 48.0
    assert e4.evaluation and e4.evaluation.wdl and e4.evaluation.wdl.draw == 0.3
    assert e4.leaf_evaluation and e4.leaf_evaluation.value == 0.2
    assert d4.principal_variation and d4.principal_variation.moves == ("d2d4", "d7d5")
    with pytest.raises(SearchCapabilityError):
        trace.require(SearchCapability.FULL_EVENTS)


@pytest.mark.parametrize(
    "lines, message",
    [
        (["1. e2e4 (P: 60.00%) unexpected", "bestmove e2e4"], "fields"),
        (["1. e2e4 (WL: 0.1)", "bestmove e2e4"], "Invalid"),
        (["1. e2e4 (P: 100.00%)", "bestmove d2d4"], "absent"),
        (["info nodes 3"], "bestmove"),
    ],
)
def test_rejects_unknown_or_incomplete_lc0_output(lines, message):
    with pytest.raises(Lc0OutputError, match=message):
        Lc0RootSnapshotParser().parse(
            lines, request=Lc0SearchRequest(ROOT_FEN, nodes=3), engine_version="test", network="test"
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({}, "exactly one"),
        ({"nodes": 1, "time_ms": 1}, "exactly one"),
        ({"nodes": -1}, "non-negative"),
        ({"time_ms": -1}, "non-negative"),
    ],
)
def test_search_request_rejects_invalid_budget(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Lc0SearchRequest(ROOT_FEN, **kwargs)


def test_parses_root_result_with_time_budget_when_move_stats_are_unavailable():
    trace = Lc0RootSnapshotParser().parse(
        ["info time 7", "bestmove e2e4"],
        request=Lc0SearchRequest(ROOT_FEN, time_ms=10),
        engine_version="test",
        network="test",
    )
    assert trace.capability is SearchCapability.ROOT_RESULT
    assert trace.snapshots[0].budget and trace.snapshots[0].budget.observed == 7
    assert trace.snapshots[0].actions is None


@pytest.mark.parametrize(
    "lines, message",
    [
        (["bestmove invalid"], "bestmove"),
        (["not-a-move (P: 100.00%)", "bestmove e2e4"], "move-stat line"),
        (["e2e4 (P: 100.00%) (PV: d2d4)", "bestmove e2e4"], "principal variation"),
        (["e2e4 (P: 1.0)", "bestmove e2e4"], "Invalid"),
    ],
)
def test_rejects_unsupported_lc0_output_shapes(lines, message):
    with pytest.raises(Lc0OutputError, match=message):
        Lc0RootSnapshotParser().parse(
            lines, request=Lc0SearchRequest(ROOT_FEN, nodes=1), engine_version="test", network="test"
        )


def test_process_adapter_runs_uci_and_parses_result(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="info nodes 2\nbestmove e2e4\n", stderr="")

    monkeypatch.setattr(lc0_adapter.subprocess, "run", fake_run)
    trace = Lc0ProcessAdapter("lc0", engine_version="v-test", network="weights").run(
        Lc0SearchRequest(ROOT_FEN, nodes=2, options={"VerboseMoveStats": True}), timeout=3
    )
    assert trace.snapshots[0].selection and trace.snapshots[0].selection.move == "e2e4"
    assert calls[0][0] == (["lc0"],)
    assert "setoption name VerboseMoveStats value true" in calls[0][1]["input"]
    assert calls[0][1]["timeout"] == 3


@pytest.mark.parametrize(
    "result, message",
    [
        (OSError("missing"), "Could not run"),
        (SimpleNamespace(returncode=2, stdout="", stderr="bad network"), "status 2"),
    ],
)
def test_process_adapter_reports_execution_failures(monkeypatch, result, message):
    def fake_run(*_args, **_kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(lc0_adapter.subprocess, "run", fake_run)
    with pytest.raises(Lc0OutputError, match=message):
        Lc0ProcessAdapter("lc0", engine_version="test", network="test").run(Lc0SearchRequest(ROOT_FEN, nodes=1))


@pytest.mark.integration
def test_optional_pinned_lc0_process_adapter():
    executable, network, version = (os.environ.get(name) for name in ("LC0_EXECUTABLE", "LC0_NETWORK", "LC0_VERSION"))
    if not all((executable, network, version)):
        pytest.skip("set LC0_EXECUTABLE, LC0_NETWORK, and LC0_VERSION for pinned lc0 adapter conformance")
    trace = Lc0ProcessAdapter(executable, engine_version=version, network=network).run(
        Lc0SearchRequest(ROOT_FEN, nodes=1, options={"WeightsFile": network, "VerboseMoveStats": True})
    )
    assert trace.capability in {SearchCapability.ROOT_RESULT, SearchCapability.ROOT_SNAPSHOTS}
