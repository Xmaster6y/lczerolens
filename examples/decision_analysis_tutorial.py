"""Executable companion for the end-to-end decision-analysis tutorial.

The tiny deterministic module is a versioned fixture, not a chess model or a
scientific result.  It makes the public evaluator, evidence, search, and
comparison boundaries runnable without downloading weights or an engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import torch
from tensordict import TensorDict
from torch import nn

from lczerolens import LczeroBoard
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
from lczerolens.reference_search import ReferenceMCTS
from lczerolens.search_trace import SearchTrace


class _TutorialFixtureModule(nn.Module):
    """Small PyTorch fixture with the same heads consumed by the public API."""

    def __init__(self) -> None:
        super().__init__()
        root = LczeroBoard()
        after_e4 = root.copy(stack=True)
        after_e4.push_uci("e2e4")
        self.register_buffer("root_e4", torch.tensor(root.encode_move(chess.Move.from_uci("e2e4"), root.turn)))
        self.register_buffer("root_d4", torch.tensor(root.encode_move(chess.Move.from_uci("d2d4"), root.turn)))
        self.register_buffer(
            "black_e5", torch.tensor(after_e4.encode_move(chess.Move.from_uci("e7e5"), after_e4.turn))
        )

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
    model: LczeroModel
    evaluator: LczeroEvaluator


def load_fixture_evaluator() -> _FixtureRuntime:
    """Load a tiny model and its concrete chess evaluator facade."""
    model = LczeroModel(_TutorialFixtureModule(), out_keys=["policy", "value"])
    return _FixtureRuntime(model, LczeroEvaluator(model))


def evaluate(model: LczeroModel, board: LczeroBoard) -> TensorDict:
    """Adapt the public model output to the singleton evaluator search contract."""
    return model(board)


@dataclass(frozen=True)
class TutorialResult:
    """Structured observations from the tutorial, with no scientific claim."""

    evaluation: Evaluation
    search: SearchTrace
    decision: DecisionAnalysis
    counterfactual: CounterfactualPair
    counterfactual_comparison: CounterfactualComparison
    variations: dict[str, LineAnalysis]


def run_tutorial() -> TutorialResult:
    """Run the documented, hermetic decision-analysis workflow."""
    board = LczeroBoard()
    runtime = load_fixture_evaluator()
    evaluation = runtime.evaluator.evaluate(board)
    search = ReferenceMCTS(c_puct=1.0).search(board, lambda position: evaluate(runtime.model, position), simulations=4)

    # The candidate lines are exact move evidence, not generated explanation.
    variations = {
        move: analyze_line(
            board,
            (chess.Move.from_uci(move),),
            MaterialAnalyzer(FactPerspective.WHITE),
            MaterialAnalyzer(FactPerspective.BLACK),
        )
        for move in (evaluation.policy.best_move.uci(), search.snapshots[-1].selection.move)
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
    return TutorialResult(evaluation, search, decision, counterfactual, comparison, variations)


if __name__ == "__main__":
    result = run_tutorial()
    print(f"policy={result.decision.policy_move} search={result.decision.search_move}")
