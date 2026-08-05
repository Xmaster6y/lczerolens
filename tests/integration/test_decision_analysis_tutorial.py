"""The maintained tutorial is an explicit non-default integration tier."""

import pytest

from examples.decision_analysis_tutorial import run_tutorial
from lczerolens.search.trace import SearchCapability


@pytest.mark.integration
def test_decision_analysis_tutorial_runs_from_its_versioned_fixture():
    result = run_tutorial()

    assert result.search.capability is SearchCapability.REPLAYABLE
    assert result.decision.actions[result.decision.policy_move].line is not None
    assert result.decision.actions[result.decision.search_move].line is not None
    assert result.counterfactual.succeeded
    assert result.counterfactual_comparison.pair is result.counterfactual
    assert result.decision.counterfactuals == (result.counterfactual_comparison,)
