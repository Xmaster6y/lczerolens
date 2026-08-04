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

Install `datasets` for dataset adapters and concept metrics, `viz` to render heatmaps, or `backends` to use the `lc0` bindings:

```bash
pip install "lczerolens[datasets]"
pip install "lczerolens[viz]"
pip install "lczerolens[backends]"
```

### Tests

After `just test-fixtures` has fetched and checksummed the lc0 fixtures, `just
tests` runs the complete fast, offline suite (unit and conformance tests) and
produces the coverage report. `just tests-unit` and `just tests-conformance`
select either tier for diagnosis. Notebook and release checks are opt-in with
`just tests-slow`; CI retains JUnit and coverage artifacts to make failures
inspectable.

### Run Models

Get the best move predicted by a model:

```python
from lczerolens import LczeroBoard, LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
board = LczeroBoard()

output = model(board)
policy = output["policy"].squeeze(0)
legal_indices = board.get_legal_indices()
best_legal_offset = policy[legal_indices].argmax()
best_move_idx = legal_indices[best_legal_offset]
print(board.decode_move(best_move_idx.item()))
```

### External Interpretability Integrations

Use `lczerolens` with your preferred PyTorch interpretability framework (`tdhook`, `captum`, `zennit`, `nnsight`). These packages are optional examples, not lczerolens abstractions or dependencies of its core evaluator contract. More examples are in the [framework-agnostic interpretability notebook](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/framework-agnostic-interpretability.html).

```python
from lczerolens import LczeroBoard, LczeroModel
from tdhook.attribution import Saliency
from tensordict import TensorDict

model = LczeroModel.from_hf("lczerolens/maia-1100")
board = LczeroBoard()

def best_logit_init_targets(td: TensorDict, _):
    policy = td["policy"]
    best_logit = policy.max(dim=-1).values
    return TensorDict(out=best_logit, batch_size=td.batch_size)

saliency_context = Saliency(init_attr_targets=best_logit_init_targets)
with saliency_context.prepare(model) as hooked_model:
    output = hooked_model(TensorDict(board=model.prepare_boards(board), batch_size=1))
    attr = output.get(("attr", "board"))
```

### Decision-analysis documentation

The maintained documentation covers the evaluator and board contract, exact
facts and move/variation evidence, constrained counterfactuals, typed search
traces, and observable behavior comparisons. Start with the [scope and
compatibility policy](https://lczerolens.readthedocs.io/en/latest/scope.html),
then follow the [facts](https://lczerolens.readthedocs.io/en/latest/facts.html),
[search](https://lczerolens.readthedocs.io/en/latest/search.html), and
[behavior](https://lczerolens.readthedocs.io/en/latest/behavior.html) guides.

Interpretability techniques remain external integrations rather than
lczerolens APIs.

### Example notebooks

The following maintained examples live in the repository and can be opened in
Colab:

- [Encode Boards](docs/source/notebooks/features/encode-boards.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/encode-boards.ipynb)
- [Load Models](docs/source/notebooks/features/load-models.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/load-models.ipynb)
- [Move Prediction](docs/source/notebooks/features/move-prediction.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/move-prediction.ipynb)
- [Run Models on GPU](docs/source/notebooks/features/run-models-on-gpu.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/run-models-on-gpu.ipynb)
- [Evaluate Models on Puzzles](docs/source/notebooks/features/evaluate-models-on-puzzles.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/evaluate-models-on-puzzles.ipynb)
- [Convert Official Weights](docs/source/notebooks/features/convert-official-weights.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/convert-official-weights.ipynb)
- [Visualise Heatmaps](docs/source/notebooks/features/visualise-heatmaps.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/visualise-heatmaps.ipynb)
- [Probe Concepts](docs/source/notebooks/features/probe-concepts.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/probe-concepts.ipynb)

### Tutorials

- [Walkthrough](docs/source/notebooks/walkthrough.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/walkthrough.ipynb)
- [Framework-Agnostic Interpretability](docs/source/notebooks/tutorials/framework-agnostic-interpretability.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/tutorials/framework-agnostic-interpretability.ipynb)

The automated-interpretability and learned-look-ahead notebooks are incomplete
and intentionally not listed as tutorials. Their techniques remain external
integrations rather than lczerolens API guarantees.

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
