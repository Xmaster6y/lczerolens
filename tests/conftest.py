"""
File to test the encodings for the Leela Chess Zero engine.
"""

import hashlib
from pathlib import Path

import onnxruntime as ort
import pytest
from lczero.backends import Backend, Weights

from lczerolens import LczeroModel
from lczerolens import backends as lczero_utils


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


@pytest.fixture(scope="session")
def tiny_lczero_backend():
    lczero_weights = Weights(str(_require_fixture("tinygyal-8.pb.gz")))
    yield Backend(weights=lczero_weights)


@pytest.fixture(scope="session")
def tiny_ensure_network():
    lczero_utils.convert_to_onnx(str(_require_fixture("tinygyal-8.pb.gz")), "assets/tinygyal-8.onnx")
    yield


@pytest.fixture(scope="session")
def tiny_model(tiny_ensure_network):
    yield LczeroModel.from_path("assets/tinygyal-8.onnx")


@pytest.fixture(scope="session")
def tiny_senet_ort(tiny_ensure_network):
    senet_ort = ort.InferenceSession("assets/tinygyal-8.onnx")
    yield senet_ort


@pytest.fixture(scope="session")
def maia_ensure_network():
    lczero_utils.convert_to_onnx("assets/maia-1100.pb.gz", "assets/maia-1100.onnx")
    yield


@pytest.fixture(scope="session")
def maia_model(maia_ensure_network):
    yield LczeroModel.from_path("assets/maia-1100.onnx")


@pytest.fixture(scope="session")
def maia_senet_ort(maia_ensure_network):
    senet_ort = ort.InferenceSession("assets/maia-1100.onnx")
    yield senet_ort


@pytest.fixture(scope="session")
def winner_ensure_network():
    lczero_utils.convert_to_onnx(
        "assets/384x30-2022_0108_1903_17_608.pb.gz",
        "assets/384x30-2022_0108_1903_17_608.onnx",
    )
    yield


@pytest.fixture(scope="session")
def winner_model(winner_ensure_network):
    yield LczeroModel.from_path("assets/384x30-2022_0108_1903_17_608.onnx")


@pytest.fixture(scope="session")
def winner_senet_ort(winner_ensure_network):
    yield ort.InferenceSession("assets/384x30-2022_0108_1903_17_608.onnx")


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
        elif "backends" in item.keywords or {"tiny_model", "winner_model", "maia_model"} & set(item.fixturenames):
            item.add_marker(pytest.mark.conformance)
        else:
            item.add_marker(pytest.mark.unit)
