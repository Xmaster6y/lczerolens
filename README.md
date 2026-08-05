<img src="https://raw.githubusercontent.com/Xmaster6y/lczerolens/refs/heads/main/docs/source/_static/images/lczerolens-logo.svg" alt="logo" width="200"/>

# lczerolens 🔍

[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://lczerolens.readthedocs.io)
[![lczerolens](https://img.shields.io/pypi/v/lczerolens?color=purple)](https://pypi.org/project/lczerolens/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/Xmaster6y/lczerolens/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![python versions](https://img.shields.io/pypi/pyversions/lczerolens.svg)](https://www.python.org/downloads/)

[![codecov](https://codecov.io/gh/Xmaster6y/lczerolens/graph/badge.svg?token=JKJAWB451A)](https://codecov.io/gh/Xmaster6y/lczerolens)
![ci-tests](https://github.com/Xmaster6y/lczerolens/actions/workflows/ci-tests.yml/badge.svg)
![ci-tests-slow](https://github.com/Xmaster6y/lczerolens/actions/workflows/ci-tests-slow.yml/badge.svg)
![publish](https://github.com/Xmaster6y/lczerolens/actions/workflows/publish.yml/badge.svg)
[![docs](https://readthedocs.org/projects/lczerolens/badge/?version=latest)](https://lczerolens.readthedocs.io/en/latest/?badge=latest)

Leela Chess Zero (lc0) Lens (`lczerolens`) makes lc0-family models portable and operable in PyTorch, then expresses their evaluator and search behavior as chess-domain evidence. It provides the model and chess-analysis boundary; interpretability methods remain external integrations. See the [scope and compatibility policy](https://lczerolens.readthedocs.io/en/latest/scope.html).

## Getting Started

### Installs

```bash
pip install lczerolens
```

Loading or publishing models through Hugging Face Hub requires the `hub` extra:

```bash
pip install "lczerolens[hub]"
```

### Tests

After `just test-fixtures` has fetched and checksummed the lc0 fixtures, `just
tests` runs the complete fast, offline suite (unit and conformance tests) and
produces the coverage report. `just tests-unit` and `just tests-conformance`
select either tier for diagnosis. The native Lczero bindings are a test-only
conformance oracle, installed through the `conformance` dependency group rather
than exposed as library API. Notebook and release checks are opt-in with `just
tests-slow`; `just tests-wheel` builds and installs the wheel in a fresh virtual
environment before running the maintained workflow. CI retains JUnit and
coverage artifacts to make failures inspectable.

### Evaluate a position

Get the best move predicted by a model:

```python
import chess
from lczerolens import LczeroEvaluator, LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
evaluator = LczeroEvaluator(model)
board = chess.Board()

evaluation = evaluator.evaluate(board)
print(evaluation.policy.best_move)
print(evaluation.policy["e2e4"].probability)
```

### External Interpretability Integrations

Use `lczerolens` with your preferred PyTorch interpretability framework
(`tdhook`, `captum`, `zennit`, or `nnsight`). These packages own their methods;
they are not lczerolens abstractions or dependencies of its evaluator contract.

```python
import chess
from lczerolens import LczeroEvaluator, LczeroKeys, LczeroModel
from tdhook.attribution import Saliency
from tensordict import TensorDict

model = LczeroModel.from_hf("lczerolens/maia-1100")
evaluator = LczeroEvaluator(model)
board = chess.Board()

def best_logit_init_targets(td: TensorDict, _):
    policy = td[LczeroKeys.NETWORK_POLICY_LOGITS]
    best_logit = policy.max(dim=-1).values
    return TensorDict(out=best_logit, batch_size=td.batch_size)

saliency_context = Saliency(init_attr_targets=best_logit_init_targets)
with saliency_context.prepare(evaluator.model) as hooked_model:
    tensors = hooked_model(evaluator.prepare([board]))
    evaluation = evaluator.finish([board], tensors)[0]
    attr = tensors.get(("attr", "input", "planes"))
```

### Define and grade a puzzle

Puzzle correctness comes from an authored solution tree rather than from model
preference or chess terminality:

```python
import chess
from lczerolens import Puzzle, PuzzleContinuation, PuzzleSolution

board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
solution = PuzzleSolution((PuzzleContinuation("g6g7"),))
puzzle = Puzzle.from_board(board, solution)

attempt = puzzle.grade(["g6g7"])
print(attempt.status)  # PuzzleStatus.SOLVED
```

Solution trees can retain alternative accepted moves and authored opponent
replies. Provider-specific dataset ingestion remains outside the core package.

### Decision-analysis documentation

The maintained documentation covers the evaluator and position contract, exact
facts and move/variation evidence, authored puzzles, constrained
counterfactuals, typed search traces, and concrete decision comparisons. Start
with the [scope and compatibility policy](https://lczerolens.readthedocs.io/en/latest/scope.html),
then follow the [facts](https://lczerolens.readthedocs.io/en/latest/facts.html),
[search](https://lczerolens.readthedocs.io/en/latest/search.html), and
[use cases](https://lczerolens.readthedocs.io/en/latest/use-cases.html) guides.

Interpretability techniques remain external integrations rather than
lczerolens APIs.

### Maintained tutorial

The executable [decision-analysis tutorial](examples/decision_analysis_tutorial.py)
composes evaluator, search, exact line analysis, and counterfactual comparison
against a deterministic fixture. Its integration test is the supported example
contract; historical notebooks built on removed APIs are not shipped.

## Full Documentation

See the full [documentation](https://lczerolens.readthedocs.io).

## Contribute

See the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If you're using `lczerolens` in your research, please cite it using the following BibTeX entry:

```bibtex
@software{poupart_lczerolens_2026,
  author = {Poupart, Yoann},
  title = {LCZeroLens},
  version = {0.4.0},
  year = {2026},
  url = {https://github.com/Xmaster6y/lczerolens}
}
```

## License

`lczerolens` is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
