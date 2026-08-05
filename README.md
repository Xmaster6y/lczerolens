<div align="center">
<img src="https://raw.githubusercontent.com/Xmaster6y/lczerolens/refs/heads/main/docs/source/_static/images/lczerolens-logo.svg" alt="logo" width="200"/>
</div>

<h1 align=center><code>lczerolens</code> 🔍</h1>

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

Portable PyTorch evaluation and chess-decision evidence for lc0-family models.

## Getting Started

Install the package. Use the `hub` extra for the example below:

```bash
pip install "lczerolens[hub]"
```

Evaluate a position:

```python
import chess
from lczerolens import LczeroEvaluator, LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
evaluator = LczeroEvaluator(model)
evaluation = evaluator.evaluate(chess.Board())

print(evaluation.policy.best_move)
print(evaluation.policy["e2e4"].probability)
```

## Examples

- [Model inputs and loading](https://lczerolens.readthedocs.io/en/latest/notebooks/features/models-and-inputs.html)
- [Evaluate positions and batches](https://lczerolens.readthedocs.io/en/latest/notebooks/features/evaluate-positions.html)
- [Chess evidence: moves, lines, puzzles, and counterfactuals](https://lczerolens.readthedocs.io/en/latest/notebooks/features/chess-evidence.html)
- [Search and replay](https://lczerolens.readthedocs.io/en/latest/notebooks/features/replayable-search.html)
- [Complete decision-analysis tutorial](examples/decision_analysis_tutorial.py)

For the package boundary and detailed guides, see the [documentation](https://lczerolens.readthedocs.io).

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md). Development and test commands live there rather than in the quick-start path.

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
