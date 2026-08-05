"""
File to test the encodings for the Leela Chess Zero engine.
"""

import hashlib
from pathlib import Path
import subprocess

import pytest
from lczero.backends import Backend, Weights

from lczerolens import LczeroModel


ROOT = Path(__file__).parents[1]
FIXTURE_MANIFEST = ROOT / "assets" / "test-fixtures.sha256"


def _require_fixture(name: str) -> Path:
    """Return a verified conformance fixture without ever downloading it."""
    fixture = ROOT / "assets" / name
    expected = {
        line.split(maxsplit=1)[1].strip(): line.split(maxsplit=1)[0]
        for line in FIXTURE_MANIFEST.read_text().splitlines()
        if line and not line.startswith("#")
    }
    if not fixture.is_file():
        pytest.fail(
            f"Missing conformance fixture {fixture}. Run `just test-fixtures` before this tier.",
            pytrace=False,
        )
    if hashlib.sha256(fixture.read_bytes()).hexdigest() != expected[name]:
        pytest.fail(f"Fixture checksum mismatch for {fixture}; re-run `just test-fixtures`.", pytrace=False)
    return fixture


def _convert_to_onnx(source: Path, destination: Path) -> None:
    """Convert a native test fixture without exposing an executable wrapper as library API."""
    subprocess.run(
        ["lc0", "leela2onnx", f"--input={source}", f"--output={destination}"],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="session")
def tiny_lczero_backend():
    lczero_weights = Weights(str(_require_fixture("tinygyal-8.pb.gz")))
    yield Backend(weights=lczero_weights)


@pytest.fixture(scope="session")
def tiny_ensure_network():
    _convert_to_onnx(_require_fixture("tinygyal-8.pb.gz"), ROOT / "assets" / "tinygyal-8.onnx")
    yield


@pytest.fixture(scope="session")
def tiny_model(tiny_ensure_network):
    yield LczeroModel.from_path("assets/tinygyal-8.onnx")


def pytest_collection_modifyitems(items):
    """Assign a default tier only when a test has no explicit primary tier."""
    primary_tiers = {"unit", "conformance", "integration", "network", "slow"}
    for item in items:
        if primary_tiers & {marker.name for marker in item.iter_markers()}:
            continue
        path = Path(str(item.fspath))
        if "integration" in path.parts:
            item.add_marker(pytest.mark.integration)
        elif "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
        elif "network" in item.keywords:
            item.add_marker(pytest.mark.network)
        elif "conformance" in item.keywords:
            item.add_marker(pytest.mark.conformance)
        elif "tiny_model" in item.fixturenames:
            item.add_marker(pytest.mark.conformance)
        else:
            item.add_marker(pytest.mark.unit)
