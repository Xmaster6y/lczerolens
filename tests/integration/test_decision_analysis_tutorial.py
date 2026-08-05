"""The maintained tutorial is an explicit non-default integration tier."""

import pytest

from examples.decision_analysis_tutorial import run_tutorial
from lczerolens.search_trace import SearchCapability


@pytest.mark.integration
def test_decision_analysis_tutorial_runs_from_its_versioned_fixture():
    result = run_tutorial()

    assert result.search.capability is SearchCapability.REPLAYABLE
    assert result.decision.evaluator_line is not None
    assert result.decision.search_line is not None
    assert result.counterfactual.succeeded
    assert result.counterfactual_behavior.targets[0].move == "e7e5"
