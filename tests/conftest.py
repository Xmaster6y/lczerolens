"""
File to test the encodings for the Leela Chess Zero engine.
"""

import pytest
from lczero.backends import Backend, Weights


@pytest.fixture(scope="session")
def lczero_backend():
    lczero_weights = Weights("ignored/BT2-768x15-swa-3250000.pb.gz")
    yield Backend(weights=lczero_weights)
