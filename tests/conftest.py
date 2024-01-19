"""
File to test the encodings for the Leela Chess Zero engine.
"""

import pytest
from lczero.backends import Backend, Weights


@pytest.fixture(scope="session")
def lczero_backend():
    lczero_weights = Weights("assets/tinygyal-8.pb.gz")
    yield Backend(weights=lczero_weights)
