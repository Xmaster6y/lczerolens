"""Regression coverage for pytest's mutually exclusive primary test tiers."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
LIVE_LC0_TEST = "tests/unit/test_lczero_search.py::test_optional_pinned_lczero_process_adapter"


def _collect(marker: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            LIVE_LC0_TEST,
            "-m",
            marker,
            "--co",
            "-q",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode in {0, 5}, result.stderr
    return result


def test_unit_collection_excludes_explicit_integration_test():
    result = _collect("unit")
    assert result.returncode == 5
    assert LIVE_LC0_TEST not in result.stdout


def test_integration_collection_includes_explicit_integration_test():
    result = _collect("integration")
    assert result.returncode == 0
    assert LIVE_LC0_TEST in result.stdout
