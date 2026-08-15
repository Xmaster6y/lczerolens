"""Optional adapter for root-level evidence emitted by the official lc0 UCI engine.

This module deliberately consumes only public process output. It does not
inspect lc0's in-memory tree, and its traces never advertise full events or
replayability.
"""

from __future__ import annotations

import math
import re
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import chess

from .capabilities import SearchAdapterCapability, require_adapter_capability
from .limits import Nodes, SearchLimit, Time
from .result import SearchResult

from .trace import (
    ChessPlayer,
    EdgeStatistics,
    PositionEvaluation,
    PrincipalVariation,
    RootAction,
    RootSelection,
    RootSnapshot,
    SearchBudget,
    SearchBudgetUnit,
    SearchCapability,
    SearchParameter,
    SearchProvenance,
    SearchTrace,
    ValuePerspective,
    Wdl,
)

if TYPE_CHECKING:
    from .reference import LeafEvaluationReplacement


class LczeroOutputError(ValueError):
    """Raised when claimed lc0 root output cannot be represented safely."""


_MOVE_STAT = re.compile(r"^\s*(?:info string\s+)?(?:\d+\.\s*)?(?P<move>[a-h][1-8][a-h][1-8][qrbn]?)(?P<fields>.*)$")
_FIELD = re.compile(
    r"(?:\((?P<name>P|Q|U|V|WL|D|PV|W|M|S|Q\+U):\s*(?P<value>[^)]*)\)|"
    r"\b(?P<bare_name>N):\s*(?P<bare_value>\d+)(?:\s*\(\+\s*\d+\))?)"
)
_MOVE_STAT_ANNOTATIONS = re.compile(r"(?:\(\s*\d+\s*\))?(?:\s*\([TWL]\))?")
_INFO_NUMBER = re.compile(r"\b(?P<name>nodes|time)\s+(?P<value>\d+)\b")


@dataclass(frozen=True)
class _LczeroSearchRequest:
    """Public UCI inputs for a single lc0 root search."""

    root_fen: str
    root_start_fen: str | None = None
    root_move_history: tuple[str, ...] | None = None
    nodes: int | None = None
    time_ms: int | None = None
    options: Mapping[str, str | int | float | bool] | None = None

    def __post_init__(self) -> None:
        if (self.nodes is None) == (self.time_ms is None):
            raise ValueError("Specify exactly one of nodes or time_ms.")
        if self.nodes is not None and self.nodes < 0:
            raise ValueError("nodes must be non-negative.")
        if self.time_ms is not None and self.time_ms < 0:
            raise ValueError("time_ms must be non-negative.")
        if (self.root_start_fen is None) != (self.root_move_history is None):
            raise ValueError("Root history requires both a starting FEN and move sequence.")
        board = self.root_board()
        if self.root_start_fen is not None and board.fen(en_passant="fen") != self.root_fen:
            raise ValueError("Root history must reconstruct root_fen.")
        if board.is_game_over():
            raise ValueError("lc0 root snapshots require a non-terminal root FEN.")

    @classmethod
    def from_board(cls, board: chess.Board, **kwargs) -> _LczeroSearchRequest:
        """Build a request that preserves every move retained by ``board``."""
        root = board.root()
        return cls(
            root_fen=board.fen(en_passant="fen"),
            root_start_fen=root.fen(en_passant="fen"),
            root_move_history=tuple(move.uci() for move in board.move_stack),
            **kwargs,
        )

    def root_board(self) -> chess.Board:
        """Reconstruct the requested root, including retained history when present."""
        board = chess.Board(self.root_start_fen or self.root_fen)
        for move in self.root_move_history or ():
            board.push_uci(move)
        return board

    @property
    def position_command(self) -> str:
        """Render the lossless UCI position command for this request."""
        if self.root_start_fen is None or self.root_move_history is None:
            return f"position fen {self.root_fen}"
        moves = "" if not self.root_move_history else f" moves {' '.join(self.root_move_history)}"
        return f"position fen {self.root_start_fen}{moves}"


@dataclass(frozen=True)
class _LczeroProcessAdapter:
    """Run an lc0 executable through UCI and parse its root-level output."""

    executable: Path | str
    engine_version: str
    network: str
    network_checksum: str | None = None

    def run(self, request: _LczeroSearchRequest, *, timeout: float = 60.0) -> SearchTrace:
        command = ["uci"]
        for name, value in (request.options or {}).items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            command.append(f"setoption name {name} value {rendered}")
        command.extend(["isready", request.position_command])
        command.append(f"go nodes {request.nodes}" if request.nodes is not None else f"go movetime {request.time_ms}")
        try:
            completed = _run_uci_process(self.executable, command, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LczeroOutputError(f"Could not run lc0 executable {self.executable!s}: {error}") from error
        if completed.returncode:
            raise LczeroOutputError(f"lc0 exited with status {completed.returncode}: {completed.stderr.strip()}")
        return _LczeroRootSnapshotParser().parse(
            completed.stdout.splitlines(),
            request=request,
            engine_version=self.engine_version,
            network=self.network,
            network_checksum=self.network_checksum,
        )


class LczeroSearch:
    """Official Lczero UCI search behind the shared ``run`` interface.

    Only public root output is translated. This adapter never claims private
    engine events, a complete tree, or replayability.
    """

    def __init__(
        self,
        *,
        executable: Path | str,
        network: str,
        engine_version: str,
        network_checksum: str | None = None,
        options: Mapping[str, str | int | float | bool] | None = None,
        timeout: float = 60.0,
    ):
        if not network:
            raise ValueError("LczeroSearch requires a network path or identifier.")
        if not engine_version:
            raise ValueError("LczeroSearch requires an explicit engine version.")
        if (
            not isinstance(timeout, int | float)
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("LczeroSearch timeout must be positive seconds.")
        self._adapter = _LczeroProcessAdapter(
            executable,
            engine_version=engine_version,
            network=network,
            network_checksum=network_checksum,
        )
        self._network = network
        self._options = dict(options or {})
        self._timeout = float(timeout)

    capabilities: frozenset[SearchAdapterCapability] = frozenset()

    def supports(self, capability: SearchAdapterCapability) -> bool:
        """Return whether this adapter implements an optional search input."""
        if not isinstance(capability, SearchAdapterCapability):
            raise TypeError("Search adapter capability checks require a SearchAdapterCapability value.")
        return capability in self.capabilities

    def require(self, capability: SearchAdapterCapability) -> LczeroSearch:
        """Require an optional input capability and return this adapter."""
        require_adapter_capability(self.capabilities, capability)
        return self

    def run(
        self,
        board: chess.Board,
        limit: SearchLimit,
        *,
        root_evaluation: LeafEvaluationReplacement | None = None,
    ) -> SearchResult:
        """Run official Lczero with a supported limit and return root evidence."""
        if root_evaluation is not None:
            self.require(SearchAdapterCapability.ROOT_EVALUATION_REPLACEMENT)
        if not isinstance(board, chess.Board):
            raise TypeError("LczeroSearch.run requires a python-chess Board.")
        if board.uci_variant != "chess" or board.chess960:
            raise ValueError("LczeroSearch currently supports standard chess positions.")
        if isinstance(limit, Nodes):
            budget = {"nodes": limit.value}
        elif isinstance(limit, Time):
            if not float(limit.milliseconds).is_integer():
                raise ValueError("LczeroSearch requires whole-millisecond Time limits.")
            budget = {"time_ms": int(limit.milliseconds)}
        else:
            raise ValueError("LczeroSearch supports only Nodes and Time limits.")
        options = {**self._options, "WeightsFile": self._network, "VerboseMoveStats": True}
        request = _LczeroSearchRequest.from_board(board, options=options, **budget)
        trace = self._adapter.run(request, timeout=self._timeout)
        return SearchResult.from_trace(trace)


class _LczeroRootSnapshotParser:
    """Parse the pinned public ``VerboseMoveStats``/UCI text contract.

    Supported fields are ``P``, ``N``, ``Q``, ``U``, ``V``, ``WL``, ``D``,
    and ``PV``. Lc0's ``(T)``, ``(W)``, and ``(L)`` proven-outcome markers
    are recognized as annotations, but are not promoted to trace evidence.
    Lines which look like a move-stat record but do not use this shape fail
    rather than silently yielding partial evidence.
    """

    format_version = "lc0-public-root-v1"

    def parse(
        self,
        output: Iterable[str],
        *,
        request: _LczeroSearchRequest,
        engine_version: str,
        network: str,
        network_checksum: str | None = None,
    ) -> SearchTrace:
        root_board = request.root_board()
        actions: list[RootAction] = []
        bestmove: str | None = None
        observed_nodes: int | None = None
        observed_time: int | None = None
        for line in (line.rstrip() for line in output):
            if line.startswith("bestmove "):
                tokens = line.split()
                if (
                    len(tokens) < 2
                    or not _is_uci(tokens[1])
                    or chess.Move.from_uci(tokens[1]) not in root_board.legal_moves
                ):
                    raise LczeroOutputError(f"Unsupported lc0 bestmove line: {line!r}")
                bestmove = tokens[1]
            elif _MOVE_STAT.match(line):
                action = self._parse_action(line)
                if chess.Move.from_uci(action.statistics.move) not in root_board.legal_moves:
                    raise LczeroOutputError(f"lc0 root statistics contained illegal move {action.statistics.move!r}.")
                actions.append(action)
            elif line.startswith("info "):
                for match in _INFO_NUMBER.finditer(line):
                    if match["name"] == "nodes":
                        observed_nodes = int(match["value"])
                    else:
                        observed_time = int(match["value"])
        if bestmove is None:
            raise LczeroOutputError("lc0 output did not contain a UCI bestmove line.")
        if actions and bestmove not in {action.statistics.move for action in actions}:
            raise LczeroOutputError("lc0 bestmove was absent from captured root action statistics.")
        actions = _normalise_priors(actions)
        unit = SearchBudgetUnit.NODES if request.nodes is not None else SearchBudgetUnit.TIME_MS
        requested = request.nodes if request.nodes is not None else request.time_ms
        observed = observed_nodes if unit is SearchBudgetUnit.NODES else observed_time
        parameters = (SearchParameter("parser_format", self.format_version),) + tuple(
            SearchParameter(f"uci.{name}", value) for name, value in (request.options or {}).items()
        )
        root_player = ChessPlayer.WHITE if root_board.turn else ChessPlayer.BLACK
        return SearchTrace(
            root_fen=request.root_fen,
            root_player=root_player,
            capability=SearchCapability.ROOT_SNAPSHOTS if actions else SearchCapability.ROOT_RESULT,
            provenance=SearchProvenance(
                "official_lc0_uci", "lc0", engine_version, network, network_checksum, parameters
            ),
            snapshots=(
                RootSnapshot(
                    0,
                    RootSelection(bestmove, "engine bestmove", "engine-defined"),
                    budget=SearchBudget(unit, requested=requested, observed=observed),
                    actions=tuple(actions) if actions else None,
                ),
            ),
            root_start_fen=request.root_start_fen,
            root_move_history=request.root_move_history,
        )

    def _parse_action(self, line: str) -> RootAction:
        match = _MOVE_STAT.match(line)
        if match is None:
            raise LczeroOutputError(f"Unsupported lc0 root move-stat line: {line!r}")
        fields = {
            field["name"] or field["bare_name"]: (field["value"] or field["bare_value"]).strip()
            for field in _FIELD.finditer(match["fields"])
        }
        remainder = _FIELD.sub("", match["fields"]).strip()
        if not fields or _MOVE_STAT_ANNOTATIONS.fullmatch(remainder) is None:
            raise LczeroOutputError(f"Unsupported lc0 root move-stat fields: {line!r}")
        try:
            prior = _number(fields["P"], percent=True) if "P" in fields else None
            visits = int(fields["N"]) if "N" in fields else None
            mean = _reported_number(fields["Q"]) if "Q" in fields else None
            exploration = _reported_number(fields["U"]) if "U" in fields else None
            total = _reported_number(fields["W"]) if "W" in fields else None
            evaluation = _evaluation(fields, "WL", "D")
            leaf_evaluation = _evaluation(fields, "V")
            pv = tuple(fields["PV"].split()) if "PV" in fields else ()
            if pv and (pv[0] != match["move"] or not all(_is_uci(move) for move in pv)):
                raise ValueError("PV must begin with the root move and contain UCI moves")
            return RootAction(
                EdgeStatistics(match["move"], ValuePerspective.ROOT_PLAYER, prior, visits, total, mean, exploration),
                evaluation=evaluation,
                leaf_evaluation=leaf_evaluation,
                principal_variation=PrincipalVariation(pv) if pv else None,
            )
        except ValueError as error:
            raise LczeroOutputError(f"Invalid lc0 root move-stat values: {line!r}") from error


def _normalise_priors(actions: list[RootAction]) -> list[RootAction]:
    """Restore the unit sum lost when lc0 prints every prior independently."""
    priors = [action.statistics.prior for action in actions]
    if not priors or any(prior is None for prior in priors):
        return actions
    total = sum(prior for prior in priors if prior is not None)
    if total <= 0:
        raise LczeroOutputError("lc0 exposed root priors with a non-positive total.")
    return [
        replace(action, statistics=replace(action.statistics, prior=action.statistics.prior / total))
        for action in actions
    ]


def _run_uci_process(
    executable: Path | str, command: list[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run one UCI search and stop the engine only after its response."""
    process = subprocess.Popen(
        [str(executable)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if (
        process.stdin is None or process.stdout is None or process.stderr is None
    ):  # pragma: no cover - subprocess contract
        process.kill()
        raise OSError("lc0 process pipes were unavailable")

    stdout: list[str] = []
    stderr: list[str] = []
    bestmove = threading.Event()
    engine_error = threading.Event()

    def read_stdout() -> None:
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            stdout.append(line)
            if line.startswith("bestmove "):
                bestmove.set()
            elif line.startswith("error "):
                engine_error.set()

    def read_stderr() -> None:
        stderr.extend(line.rstrip("\r\n") for line in process.stderr)

    readers = (
        threading.Thread(target=read_stdout, daemon=True),
        threading.Thread(target=read_stderr, daemon=True),
    )
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout
    try:
        process.stdin.write("\n".join(command) + "\n")
        process.stdin.flush()
        while not bestmove.is_set() and not engine_error.is_set() and process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired([str(executable)], timeout)
            bestmove.wait(min(remaining, 0.05))
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
        process.stdin.close()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired([str(executable)], timeout)
        returncode = process.wait(timeout=remaining)
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=1)

    if engine_error.is_set():
        message = next(line for line in stdout if line.startswith("error "))
        returncode = returncode or 1
        stderr.append(message)
    return subprocess.CompletedProcess([str(executable)], returncode, "\n".join(stdout), "\n".join(stderr))


def _number(value: str, *, percent: bool = False) -> float:
    if percent:
        if not value.endswith("%"):
            raise ValueError("P must use percent notation")
        return float(value[:-1]) / 100
    return float(value)


def _reported_number(value: str) -> float | None:
    """Return ``None`` for lc0's dashed placeholder for an unreported value."""
    if re.fullmatch(r"-\.-+", value):
        return None
    return _number(value)


def _evaluation(fields: Mapping[str, str], value_name: str, draw_name: str | None = None) -> PositionEvaluation | None:
    if value_name not in fields:
        return None
    value = _reported_number(fields[value_name])
    if value is None:
        if draw_name is not None and draw_name in fields and _reported_number(fields[draw_name]) is not None:
            raise ValueError(f"{value_name} and {draw_name} must both be reported")
        return None
    if draw_name is None:
        return PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=value)
    if draw_name not in fields:
        raise ValueError(f"{value_name} requires {draw_name}")
    draw = _reported_number(fields[draw_name])
    if draw is None:
        raise ValueError(f"{value_name} and {draw_name} must both be reported")
    return PositionEvaluation(
        ValuePerspective.ROOT_PLAYER,
        value=value,
        wdl=Wdl.from_win_loss_draw(value, draw, ValuePerspective.ROOT_PLAYER),
    )


def _is_uci(move: str) -> bool:
    return re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) is not None


__all__ = ["LczeroOutputError", "LczeroSearch"]
