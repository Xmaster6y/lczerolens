import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from lczerolens.search import lczero as lczero_adapter
from lczerolens.search.lczero import (
    LczeroOutputError,
    _LczeroProcessAdapter,
    _LczeroRootSnapshotParser,
    _LczeroSearchRequest,
)
from lczerolens.search.trace import SearchCapability, SearchCapabilityError


ROOT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_parses_versioned_lc0_root_snapshot_fixture():
    fixture = Path(__file__).parents[1] / "assets" / "lc0_root_snapshot_v1.txt"
    trace = _LczeroRootSnapshotParser().parse(
        fixture.read_text().splitlines(),
        request=_LczeroSearchRequest(ROOT_FEN, nodes=256, options={"VerboseMoveStats": True}),
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
    assert e4.statistics.total_value is None
    assert e4.evaluation and e4.evaluation.wdl and e4.evaluation.wdl.draw == 0.3
    assert e4.leaf_evaluation and e4.leaf_evaluation.value == 0.2
    assert d4.principal_variation is None
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
    with pytest.raises(LczeroOutputError, match=message):
        _LczeroRootSnapshotParser().parse(
            lines, request=_LczeroSearchRequest(ROOT_FEN, nodes=3), engine_version="test", network="test"
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
        _LczeroSearchRequest(ROOT_FEN, **kwargs)


def test_search_request_rejects_partial_or_inconsistent_retained_history():
    with pytest.raises(ValueError, match="both a starting FEN"):
        _LczeroSearchRequest(ROOT_FEN, root_start_fen=ROOT_FEN, nodes=1)
    with pytest.raises(ValueError, match="reconstruct"):
        _LczeroSearchRequest(ROOT_FEN, root_start_fen=ROOT_FEN, root_move_history=("e2e4",), nodes=1)


def test_search_request_rejects_terminal_root_before_invoking_lc0():
    with pytest.raises(ValueError, match="non-terminal"):
        _LczeroSearchRequest("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", nodes=1)


def test_parses_root_result_with_time_budget_when_move_stats_are_unavailable():
    trace = _LczeroRootSnapshotParser().parse(
        ["info time 7", "bestmove e2e4"],
        request=_LczeroSearchRequest(ROOT_FEN, time_ms=10),
        engine_version="test",
        network="test",
    )
    assert trace.capability is SearchCapability.ROOT_RESULT
    assert trace.snapshots[0].budget and trace.snapshots[0].budget.observed == 7
    assert trace.snapshots[0].actions is None


def test_normalises_rounded_priors_and_preserves_only_reported_total_value():
    trace = _LczeroRootSnapshotParser().parse(
        [
            "info string e2e4 N: 1 (P: 33.33%) (Q: 0.0) (W: 0.0)",
            "info string d2d4 N: 1 (P: 33.33%) (Q: 0.0) (W: 0.0)",
            "info string g1f3 N: 1 (P: 33.33%) (Q: 0.0) (W: 0.0)",
            "bestmove e2e4",
        ],
        request=_LczeroSearchRequest(ROOT_FEN, nodes=3),
        engine_version="test",
        network="test",
    )
    actions = trace.snapshots[0].actions or ()
    assert sum(action.statistics.prior or 0.0 for action in actions) == pytest.approx(1.0)
    assert all(action.statistics.total_value == 0.0 for action in actions)


def test_preserves_unreported_lczero_values_as_missing():
    trace = _LczeroRootSnapshotParser().parse(
        [
            "info string e2e4 (322) N: 0 (+ 0) (P: 100.00%) (WL: -.-----) (D: -.---) "
            "(Q: 0.11550) (U: 0.74239) (V: -.----)",
            "bestmove e2e4",
        ],
        request=_LczeroSearchRequest(ROOT_FEN, nodes=1),
        engine_version="test",
        network="test",
    )
    action = (trace.snapshots[0].actions or ())[0]
    assert action.statistics.mean_value == pytest.approx(0.1155)
    assert action.evaluation is None
    assert action.leaf_evaluation is None


def test_rejects_non_positive_priors_and_non_move_stat_records():
    parser = _LczeroRootSnapshotParser()
    with pytest.raises(LczeroOutputError, match="non-positive"):
        parser.parse(
            ["e2e4 (P: 0.00%)", "bestmove e2e4"],
            request=_LczeroSearchRequest(ROOT_FEN, nodes=1),
            engine_version="test",
            network="test",
        )
    with pytest.raises(LczeroOutputError, match="move-stat line"):
        parser._parse_action("not a move stat")


@pytest.mark.parametrize(
    "lines, message",
    [
        (["bestmove invalid"], "bestmove"),
        (["bestmove e7e5"], "bestmove"),
        (["e7e5 (P: 100.00%)", "bestmove e2e4"], "illegal move"),
        (["e2e4 (P: 100.00%) unexpected", "bestmove e2e4"], "fields"),
        (["e2e4 (P: 100.00%) (PV: d2d4)", "bestmove e2e4"], "Invalid"),
        (["e2e4 (P: 1.0)", "bestmove e2e4"], "Invalid"),
        (["e2e4 (P: 100.00%) (V: .)", "bestmove e2e4"], "Invalid"),
    ],
)
def test_rejects_unsupported_lc0_output_shapes(lines, message):
    with pytest.raises(LczeroOutputError, match=message):
        _LczeroRootSnapshotParser().parse(
            lines, request=_LczeroSearchRequest(ROOT_FEN, nodes=1), engine_version="test", network="test"
        )


def test_process_adapter_runs_uci_and_parses_result(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="info nodes 2\nbestmove e2e4\n", stderr="")

    monkeypatch.setattr(lczero_adapter, "_run_uci_process", fake_run)
    trace = _LczeroProcessAdapter("lc0", engine_version="v-test", network="weights").run(
        _LczeroSearchRequest(ROOT_FEN, nodes=2, options={"VerboseMoveStats": True}), timeout=3
    )
    assert trace.snapshots[0].selection and trace.snapshots[0].selection.move == "e2e4"
    assert calls[0][0][0] == "lc0"
    assert "setoption name VerboseMoveStats value true" in calls[0][0][1]
    assert "quit" not in calls[0][0][1]
    assert calls[0][1]["timeout"] == 3


def test_uci_process_waits_for_bestmove_before_quitting(tmp_path):
    executable = tmp_path / "fake-lc0"
    executable.write_text(
        """#!/usr/bin/env python3
import sys

for command in sys.stdin:
    command = command.strip()
    if command.startswith("go "):
        print("info nodes 1", flush=True)
        print("bestmove e2e4", flush=True)
    elif command == "quit":
        print("quit received", flush=True)
        break
"""
    )
    executable.chmod(0o755)

    completed = lczero_adapter._run_uci_process(executable, ["uci", "go nodes 1"], timeout=10)

    assert completed.returncode == 0
    assert completed.stdout.splitlines()[-2:] == ["bestmove e2e4", "quit received"]


def test_uci_process_surfaces_engine_errors(tmp_path):
    executable = tmp_path / "fake-lc0"
    executable.write_text(
        """#!/usr/bin/env python3
import sys

for command in sys.stdin:
    if command.startswith("go "):
        print("error invalid backend", flush=True)
    elif command.strip() == "quit":
        break
"""
    )
    executable.chmod(0o755)

    completed = lczero_adapter._run_uci_process(executable, ["go nodes 1"], timeout=10)

    assert completed.returncode == 1
    assert completed.stderr == "error invalid backend"


def test_uci_process_enforces_search_timeout(tmp_path):
    executable = tmp_path / "fake-lc0"
    executable.write_text(
        """#!/usr/bin/env python3
import sys
import time

for command in sys.stdin:
    if command.startswith("go "):
        time.sleep(10)
"""
    )
    executable.chmod(0o755)

    with pytest.raises(subprocess.TimeoutExpired):
        lczero_adapter._run_uci_process(executable, ["go nodes 1"], timeout=0.05)


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

    monkeypatch.setattr(lczero_adapter, "_run_uci_process", fake_run)
    with pytest.raises(LczeroOutputError, match=message):
        _LczeroProcessAdapter("lc0", engine_version="test", network="test").run(
            _LczeroSearchRequest(ROOT_FEN, nodes=1)
        )


@pytest.mark.integration
def test_optional_pinned_lczero_process_adapter():
    executable, network, version = (os.environ.get(name) for name in ("LC0_EXECUTABLE", "LC0_NETWORK", "LC0_VERSION"))
    if not all((executable, network, version)):
        pytest.skip("set LC0_EXECUTABLE, LC0_NETWORK, and LC0_VERSION for pinned lc0 adapter conformance")
    executable_path = Path(executable)
    network_path = Path(network)
    assert executable_path.is_file(), "LC0_EXECUTABLE must name the pinned lc0 binary"
    assert network_path.is_file(), "LC0_NETWORK must name the pinned network file"
    reported_version = subprocess.run(
        [str(executable_path), "--version"],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert version in f"{reported_version.stdout}\n{reported_version.stderr}", (
        "LC0_VERSION must match the version reported by LC0_EXECUTABLE --version"
    )
    with network_path.open("rb") as network_file:
        checksum = f"sha256:{hashlib.file_digest(network_file, 'sha256').hexdigest()}"
    backend = os.environ.get("LC0_BACKEND", "eigen")
    trace = _LczeroProcessAdapter(
        executable_path,
        engine_version=version,
        network=str(network_path),
        network_checksum=checksum,
    ).run(
        _LczeroSearchRequest(
            ROOT_FEN,
            nodes=1,
            options={"Backend": backend, "WeightsFile": network, "VerboseMoveStats": True},
        )
    )
    assert trace.capability in {SearchCapability.ROOT_RESULT, SearchCapability.ROOT_SNAPSHOTS}
    assert trace.provenance.engine_version == version
    assert trace.provenance.network_checksum == checksum
