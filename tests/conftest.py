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
