"""Regression coverage for pytest's mutually exclusive primary test tiers."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
LIVE_LC0_TEST = "tests/unit/test_lc0_adapter.py::test_optional_pinned_lc0_process_adapter"


def _collect(marker: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-m", marker, "--co", "-q"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_unit_collection_excludes_explicit_integration_test():
    assert LIVE_LC0_TEST not in _collect("unit")


def test_integration_collection_includes_explicit_integration_test():
    assert LIVE_LC0_TEST in _collect("integration")
