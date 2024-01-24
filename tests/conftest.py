"""
File to test the encodings for the Leela Chess Zero engine.
"""

import pytest
from lczero.backends import Backend, Weights

from lczerolens.utils import lczero as lczero_utils


@pytest.fixture(scope="session")
def lczero_backend():
    lczero_weights = Weights("assets/tinygyal-8.pb.gz")
    yield Backend(weights=lczero_weights)


@pytest.fixture(scope="session")
def ensure_network():
    lczero_utils.convert_to_onnx(
        "assets/tinygyal-8.pb.gz", "assets/tinygyal-8.onnx"
    )
    yield
