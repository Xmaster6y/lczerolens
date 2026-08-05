"""Executable companion for the end-to-end decision-analysis tutorial.

The tiny deterministic module is a versioned fixture, not a chess model or a
scientific result.  It makes the public evaluator, evidence, search, and
comparison boundaries runnable without downloading weights or an engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import sys

import chess
import torch
from torch import nn

from lczerolens import ReferenceSearch, SearchResult, Simulations
from lczerolens._codec import encode_move
from lczerolens.decision import (
    CounterfactualComparison,
    DecisionAnalysis,
    compare_counterfactual,
    compare_decision,
)
from lczerolens.counterfactuals import CounterfactualPair, sibling_counterfactual
from lczerolens.evaluation import Evaluation
from lczerolens.evaluator import LczeroEvaluator
from lczerolens.facts import FactPerspective, MaterialAnalyzer
from lczerolens.model import LczeroModel
from lczerolens.moves import LineAnalysis, analyze_line
from lczerolens.provenance import EvaluationProvenance


TUTORIAL_DECISION_DIGEST = "8a6646663724282bd88cb89ac303ce80702aa4ac46a728b99be002e12550b49c"


class _TutorialFixtureModule(nn.Module):
    """Small PyTorch fixture with the same heads consumed by the public API."""

    def __init__(self) -> None:
        super().__init__()
        root = chess.Board()
        after_e4 = root.copy(stack=True)
        after_e4.push_uci("e2e4")
        self.register_buffer("root_e4", torch.tensor(encode_move(root, chess.Move.from_uci("e2e4"))))
        self.register_buffer("root_d4", torch.tensor(encode_move(root, chess.Move.from_uci("d2d4"))))
        self.register_buffer("black_e5", torch.tensor(encode_move(after_e4, chess.Move.from_uci("e7e5"))))

    def forward(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = board.shape[0]
        policy = torch.zeros((batch, 1858), device=board.device)
        # A deliberately transparent position-dependent score.  It is only a
        # fixture mechanism, not an interpretation of chess features.
        file_signal = (board * torch.arange(8, device=board.device).view(1, 1, 1, 8)).sum((1, 2, 3))
        policy[:, self.root_e4] = 4.0
        policy[:, self.root_d4] = 2.0
        policy[:, self.black_e5] = file_signal
        return policy, torch.full((batch,), 0.2, device=board.device)


@dataclass(frozen=True)
class _FixtureRuntime:
    evaluator: LczeroEvaluator


def load_fixture_evaluator() -> _FixtureRuntime:
    """Load a tiny model and its concrete chess evaluator facade."""
    model = LczeroModel(_TutorialFixtureModule(), out_keys=["policy", "value"])
    provenance = EvaluationProvenance(
        source="lczerolens-tutorial-fixture",
        model_type="lczerolens.tutorial.fixture-v1",
        network="decision-analysis-tutorial-v1",
    )
    return _FixtureRuntime(LczeroEvaluator(model, provenance=provenance))


@dataclass(frozen=True)
class TutorialResult:
    """Structured observations from the tutorial, with no scientific claim."""

    evaluation: Evaluation
    search: SearchResult
    decision: DecisionAnalysis
    counterfactual: CounterfactualPair
    counterfactual_comparison: CounterfactualComparison
    variations: dict[str, LineAnalysis]
    restored_decision: DecisionAnalysis
    decision_digest: str


def run_tutorial(artifact_path: str | PathLike[str] | None = None) -> TutorialResult:
    """Run the documented, hermetic decision-analysis workflow."""
    board = chess.Board()
    runtime = load_fixture_evaluator()
    evaluation = runtime.evaluator.evaluate(board)
    search = ReferenceSearch(runtime.evaluator, c_puct=1.0).run(board, Simulations(4))

    # The candidate lines are exact move evidence, not generated explanation.
    variations = {
        move: analyze_line(
            board,
            (chess.Move.from_uci(move),),
            MaterialAnalyzer(FactPerspective.WHITE),
            MaterialAnalyzer(FactPerspective.BLACK),
        )
        for move in (evaluation.policy.best_move.uci(), search.move.uci())
    }
    counterfactual = sibling_counterfactual(board, factual="g1f3", alternative="b1c3")
    if not counterfactual.succeeded or counterfactual.alternative is None:
        raise RuntimeError("The tutorial's legal sibling counterfactual unexpectedly failed.")
    comparison = compare_counterfactual(counterfactual, runtime.evaluator)
    decision = compare_decision(
        evaluation,
        search,
        line_analyses=variations,
        counterfactuals=(comparison,),
    )
    restored = DecisionAnalysis.from_bytes(decision.to_bytes())
    if artifact_path is not None:
        decision.save(artifact_path)
        restored = DecisionAnalysis.load(artifact_path)
    if restored.digest() != decision.digest():
        raise RuntimeError("The tutorial decision artifact did not retain its canonical digest.")
    if decision.digest() != TUTORIAL_DECISION_DIGEST:
        raise RuntimeError("The versioned tutorial decision digest changed unexpectedly.")
    return TutorialResult(
        evaluation,
        search,
        decision,
        counterfactual,
        comparison,
        variations,
        restored,
        decision.digest(),
    )


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: decision_analysis_tutorial.py [artifact-path]")
    artifact = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    result = run_tutorial(artifact)
    print(f"policy={result.decision.policy_move} search={result.decision.search_move} digest={result.decision_digest}")
