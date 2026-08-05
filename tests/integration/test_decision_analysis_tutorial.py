"""The maintained tutorial is an explicit non-default integration tier."""

import pytest

from examples.decision_analysis_tutorial import TUTORIAL_DECISION_DIGEST, run_tutorial
from lczerolens import PuzzleStatus
from lczerolens.search.trace import SearchCapability


@pytest.mark.integration
def test_decision_analysis_tutorial_runs_from_its_versioned_fixture(tmp_path):
    artifact = tmp_path / "decision.json"
    result = run_tutorial(artifact)

    assert result.search.capability is SearchCapability.REPLAYABLE
    assert result.decision.actions[result.decision.policy_move].line is not None
    assert result.decision.actions[result.decision.search_move].line is not None
    assert result.counterfactual.succeeded
    assert result.counterfactual_comparison.pair is result.counterfactual
    assert result.decision.counterfactuals == (result.counterfactual_comparison,)
    assert result.puzzle_attempt.puzzle is result.puzzle
    assert result.puzzle_attempt.status is PuzzleStatus.SOLVED
    assert artifact.read_bytes() == result.decision.to_bytes()
    assert result.restored_decision == result.decision
    assert result.restored_decision.digest() == result.decision_digest
    assert result.decision_digest == TUTORIAL_DECISION_DIGEST
