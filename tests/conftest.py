"""
File to test the encodings for the Leela Chess Zero engine.
"""

import onnxruntime as ort
import pytest
from lczero.backends import Backend, Weights

from lczerolens import LczeroModel
from lczerolens import backends as lczero_utils


@pytest.fixture(scope="session")
def tiny_lczero_backend():
    lczero_weights = Weights("assets/tinygyal-8.pb.gz")
    yield Backend(weights=lczero_weights)


@pytest.fixture(scope="session")
def tiny_ensure_network():
    lczero_utils.convert_to_onnx("assets/tinygyal-8.pb.gz", "assets/tinygyal-8.onnx")
    yield


@pytest.fixture(scope="session")
def tiny_model(tiny_ensure_network):
    yield LczeroModel.from_path("assets/tinygyal-8.onnx")


@pytest.fixture(scope="session")
def tiny_senet_ort(tiny_ensure_network):
    senet_ort = ort.InferenceSession("assets/tinygyal-8.onnx")
    yield senet_ort


@pytest.fixture(scope="class")
def maia_ensure_network():
    lczero_utils.convert_to_onnx("assets/maia-1100.pb.gz", "assets/maia-1100.onnx")
    yield


@pytest.fixture(scope="class")
def maia_model(maia_ensure_network):
    yield LczeroModel.from_path("assets/maia-1100.onnx")


@pytest.fixture(scope="class")
def maia_senet_ort(maia_ensure_network):
    senet_ort = ort.InferenceSession("assets/maia-1100.onnx")
    yield senet_ort


@pytest.fixture(scope="class")
def winner_ensure_network():
    lczero_utils.convert_to_onnx(
        "assets/384x30-2022_0108_1903_17_608.pb.gz",
        "assets/384x30-2022_0108_1903_17_608.onnx",
    )
    yield


@pytest.fixture(scope="class")
def winner_model(winner_ensure_network):
    yield LczeroModel.from_path("assets/384x30-2022_0108_1903_17_608.onnx")


@pytest.fixture(scope="class")
def winner_senet_ort(winner_ensure_network):
    yield ort.InferenceSession("assets/384x30-2022_0108_1903_17_608.onnx")


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--run-fast", action="store_true", default=False, help="run fast tests")
    parser.addoption("--run-backends", action="store_true", default=False, help="run backends tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "backends: mark test as backends test")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    run_fast = config.getoption("--run-fast")
    run_backends = config.getoption("--run-backends")

    skip_slow = pytest.mark.skip(reason="--run-slow not given in cli: skipping slow tests")
    skip_fast = pytest.mark.skip(reason="--run-fast not given in cli: skipping fast tests")
    skip_backends = pytest.mark.skip(reason="--run-backends not given in cli: skipping backends tests")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "fast" in item.keywords and not run_fast:
            item.add_marker(skip_fast)
        if "backends" in item.keywords and not run_backends:
            item.add_marker(skip_backends)
