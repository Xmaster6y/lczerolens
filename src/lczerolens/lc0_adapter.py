"""Optional adapter for root-level evidence emitted by the official lc0 UCI engine.

This module deliberately consumes only public process output. It does not
inspect lc0's in-memory tree, and its traces never advertise full events or
replayability.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import chess

from .search_trace import (
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


class Lc0OutputError(ValueError):
    """Raised when claimed lc0 root output cannot be represented safely."""


_MOVE_STAT = re.compile(r"^\s*(?:\d+\.\s*)?(?P<move>[a-h][1-8][a-h][1-8][qrbn]?)\s+(?P<fields>.*)$")
_FIELD = re.compile(r"\((?P<name>P|N|Q|U|V|WL|D|PV):\s*(?P<value>[^)]*)\)")
_INFO_NUMBER = re.compile(r"\b(?P<name>nodes|time)\s+(?P<value>\d+)\b")


@dataclass(frozen=True)
class Lc0SearchRequest:
    """Public UCI inputs for a single lc0 root search."""

    root_fen: str
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
        chess.Board(self.root_fen)


@dataclass(frozen=True)
class Lc0ProcessAdapter:
    """Run an lc0 executable through UCI and parse its root-level output."""

    executable: Path | str
    engine_version: str
    network: str
    network_checksum: str | None = None

    def run(self, request: Lc0SearchRequest, *, timeout: float = 60.0) -> SearchTrace:
        command = ["uci"]
        for name, value in (request.options or {}).items():
            rendered = str(value).lower() if isinstance(value, bool) else value
            command.append(f"setoption name {name} value {rendered}")
        command.extend(["isready", f"position fen {request.root_fen}"])
        command.append(f"go nodes {request.nodes}" if request.nodes is not None else f"go movetime {request.time_ms}")
        command.append("quit")
        try:
            completed = subprocess.run(
                [str(self.executable)],
                input="\n".join(command) + "\n",
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Lc0OutputError(f"Could not run lc0 executable {self.executable!s}: {error}") from error
        if completed.returncode:
            raise Lc0OutputError(f"lc0 exited with status {completed.returncode}: {completed.stderr.strip()}")
        return Lc0RootSnapshotParser().parse(
            completed.stdout.splitlines(),
            request=request,
            engine_version=self.engine_version,
            network=self.network,
            network_checksum=self.network_checksum,
        )


class Lc0RootSnapshotParser:
    """Parse the pinned public ``VerboseMoveStats``/UCI text contract.

    Supported fields are ``P``, ``N``, ``Q``, ``U``, ``V``, ``WL``, ``D``,
    and ``PV``. Lines which look like a move-stat record but do not use this
    shape fail rather than silently yielding partial evidence.
    """

    format_version = "lc0-public-root-v1"

    def parse(
        self,
        output: Iterable[str],
        *,
        request: Lc0SearchRequest,
        engine_version: str,
        network: str,
        network_checksum: str | None = None,
    ) -> SearchTrace:
        actions: list[RootAction] = []
        bestmove: str | None = None
        observed_nodes: int | None = None
        observed_time: int | None = None
        for line in (line.rstrip() for line in output):
            if line.startswith("bestmove "):
                tokens = line.split()
                if len(tokens) < 2 or not _is_uci(tokens[1]):
                    raise Lc0OutputError(f"Unsupported lc0 bestmove line: {line!r}")
                bestmove = tokens[1]
            elif line.startswith("info "):
                for match in _INFO_NUMBER.finditer(line):
                    if match["name"] == "nodes":
                        observed_nodes = int(match["value"])
                    else:
                        observed_time = int(match["value"])
            elif any(f"({name}:" in line for name in ("P", "N", "Q", "U", "V", "WL", "D", "PV")):
                actions.append(self._parse_action(line))
        if bestmove is None:
            raise Lc0OutputError("lc0 output did not contain a UCI bestmove line.")
        if actions and bestmove not in {action.statistics.move for action in actions}:
            raise Lc0OutputError("lc0 bestmove was absent from captured root action statistics.")
        unit = SearchBudgetUnit.NODES if request.nodes is not None else SearchBudgetUnit.TIME_MS
        requested = request.nodes if request.nodes is not None else request.time_ms
        observed = observed_nodes if unit is SearchBudgetUnit.NODES else observed_time
        parameters = (SearchParameter("parser_format", self.format_version),) + tuple(
            SearchParameter(f"uci.{name}", value) for name, value in (request.options or {}).items()
        )
        root_player = ChessPlayer.WHITE if chess.Board(request.root_fen).turn else ChessPlayer.BLACK
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
        )

    def _parse_action(self, line: str) -> RootAction:
        match = _MOVE_STAT.match(line)
        if match is None:
            raise Lc0OutputError(f"Unsupported lc0 root move-stat line: {line!r}")
        fields = {field["name"]: field["value"].strip() for field in _FIELD.finditer(match["fields"])}
        if not fields or _FIELD.sub("", match["fields"]).strip():
            raise Lc0OutputError(f"Unsupported lc0 root move-stat fields: {line!r}")
        try:
            prior = _number(fields["P"], percent=True) if "P" in fields else None
            visits = int(fields["N"]) if "N" in fields else None
            mean = _number(fields["Q"]) if "Q" in fields else None
            exploration = _number(fields["U"]) if "U" in fields else None
            evaluation = _evaluation(fields, "WL", "D")
            leaf_evaluation = _evaluation(fields, "V")
        except ValueError as error:
            raise Lc0OutputError(f"Invalid lc0 root move-stat values: {line!r}") from error
        pv = tuple(fields["PV"].split()) if "PV" in fields else ()
        if pv and (pv[0] != match["move"] or not all(_is_uci(move) for move in pv)):
            raise Lc0OutputError(f"Invalid lc0 principal variation: {line!r}")
        total = mean * visits if mean is not None and visits is not None else None
        return RootAction(
            EdgeStatistics(match["move"], ValuePerspective.ROOT_PLAYER, prior, visits, total, mean, exploration),
            evaluation=evaluation,
            leaf_evaluation=leaf_evaluation,
            principal_variation=PrincipalVariation(pv) if pv else None,
        )


def _number(value: str, *, percent: bool = False) -> float:
    if percent:
        if not value.endswith("%"):
            raise ValueError("P must use percent notation")
        return float(value[:-1]) / 100
    return float(value)


def _evaluation(fields: Mapping[str, str], value_name: str, draw_name: str | None = None) -> PositionEvaluation | None:
    if value_name not in fields:
        return None
    value = _number(fields[value_name])
    if draw_name is None:
        return PositionEvaluation(ValuePerspective.ROOT_PLAYER, value=value)
    if draw_name not in fields:
        raise ValueError(f"{value_name} requires {draw_name}")
    draw = _number(fields[draw_name])
    return PositionEvaluation(
        ValuePerspective.ROOT_PLAYER,
        value=value,
        wdl=Wdl.from_win_loss_draw(value, draw, ValuePerspective.ROOT_PLAYER),
    )


def _is_uci(move: str) -> bool:
    return re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) is not None


__all__ = ["Lc0OutputError", "Lc0ProcessAdapter", "Lc0RootSnapshotParser", "Lc0SearchRequest"]
