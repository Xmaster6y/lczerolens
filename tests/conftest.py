"""
File to test the encodings for the Leela Chess Zero engine.
"""

import pytest
from lczero.backends import Backend, Weights

from lczerolens.adapt import ModelWrapper, SeNet
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


@pytest.fixture(scope="session")
def tiny_wrapper(ensure_network):
    wrapper = ModelWrapper.from_path("assets/tinygyal-8.onnx")
    yield wrapper


@pytest.fixture(scope="session")
def tiny_senet(tiny_wrapper):
    senet = SeNet(2, 16, n_hidden_red=16, heads=["policy", "value"])
    state_dict = tiny_wrapper.model.state_dict()
    new_state_dict = senet.state_dict_mapper(state_dict)
    senet.load_state_dict(new_state_dict)
    yield senet
