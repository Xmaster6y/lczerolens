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
from lczerolens.behavior import (
    CounterfactualBehaviorComparison,
    DecisionComparison,
    EvaluatorBehavior,
    compare_counterfactual_behavior,
    compare_search_decision,
    evaluator_behavior,
)
from lczerolens.counterfactuals import CounterfactualResult, sibling_counterfactual
from lczerolens.facts import FactPerspective, MaterialAnalyzer
from lczerolens.model import LczeroModel
from lczerolens.move_evidence import VariationEvidence, analyze_variation
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


def load_fixture_evaluator() -> LczeroModel:
    """Load a tiny PyTorch evaluator through the standard TensorDict wrapper."""
    return LczeroModel(_TutorialFixtureModule(), out_keys=["policy", "value"])


def evaluate(model: LczeroModel, board: LczeroBoard) -> TensorDict:
    """Adapt the public model output to the singleton evaluator search contract."""
    return model(board)


@dataclass(frozen=True)
class TutorialResult:
    """Structured observations from the tutorial, with no scientific claim."""

    evaluator: EvaluatorBehavior
    search: SearchTrace
    decision: DecisionComparison
    counterfactual: CounterfactualResult
    counterfactual_behavior: CounterfactualBehaviorComparison
    variations: dict[str, VariationEvidence]


def run_tutorial() -> TutorialResult:
    """Run the documented, hermetic decision-analysis workflow."""
    board = LczeroBoard()
    model = load_fixture_evaluator()
    evaluator = evaluator_behavior(board, evaluate(model, board))
    search = ReferenceMCTS(c_puct=1.0).search(board, lambda position: evaluate(model, position), simulations=4)

    # The candidate lines are exact move evidence, not generated explanation.
    variations = {
        move: analyze_variation(
            board,
            (chess.Move.from_uci(move),),
            MaterialAnalyzer(FactPerspective.WHITE),
            MaterialAnalyzer(FactPerspective.BLACK),
        )
        for move in (evaluator.selected_move, search.snapshots[-1].selection.move)
    }
    decision = compare_search_decision(evaluator, search, variation_evidence=variations)

    counterfactual = sibling_counterfactual(board, chess.Move.from_uci("g1f3"), chess.Move.from_uci("b1c3"))
    if not counterfactual.succeeded or counterfactual.modified is None:
        raise RuntimeError("The tutorial's legal sibling counterfactual unexpectedly failed.")
    original_board = board.copy(stack=True)
    original_board.push_uci("g1f3")
    modified_board = board.copy(stack=True)
    modified_board.push_uci("b1c3")
    original = evaluator_behavior(original_board, evaluate(model, original_board))
    modified = evaluator_behavior(modified_board, evaluate(model, modified_board))
    comparison = compare_counterfactual_behavior(
        original,
        modified,
        ("e7e5",),
        counterfactual=counterfactual,
        variation_evidence={
            "e7e5": analyze_variation(
                original_board,
                (chess.Move.from_uci("e7e5"),),
                MaterialAnalyzer(FactPerspective.WHITE),
            )
        },
    )
    return TutorialResult(evaluator, search, decision, counterfactual, comparison, variations)


if __name__ == "__main__":
    result = run_tutorial()
    print(f"evaluator={result.evaluator.selected_move} search={result.decision.search_candidate}")
