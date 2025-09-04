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

Leela Chess Zero (lc0) Lens (`lczerolens`): a set of utilities to make interpretability easy and framework-agnostic (PyTorch): use it with `tdhook`, `captum`, `zennit`, or `nnsight`.

## Getting Started

### Installs

```bash
pip install lczerolens
```

Take the `viz` extra to render heatmaps and the `backends` extra to use the `lc0` backends.

### Run Models

Get the best move predicted by a model:

```python
from lczerolens import LczeroBoard, LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
board = LczeroBoard()

output = model(board)
best_move_idx = output["policy"].argmax()
print(board.decode_move(best_move_idx))
```

### Framework-Agnostic Interpretability

Use `lczerolens` with your preferred PyTorch interpretability framework (`tdhook`, `captum`, `zennit`, `nnsight`). More examples in the [framework-agnostic interpretability notebook](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/framework-agnostic-interpretability.html).

```python
from lczerolens import LczeroBoard, LczeroModel
model = LczeroModel.from_hf("lczerolens/maia-1100")
board = LczeroBoard()

# TODO: complete this example
```

### Features

- [Encode Boards](https://lczerolens.readthedocs.io/en/latest/notebooks/features/encode-boards.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/encode-boards.ipynb)
- [Load Models](https://lczerolens.readthedocs.io/en/latest/notebooks/features/load-models.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/load-models.ipynb)
- [Move Prediction](https://lczerolens.readthedocs.io/en/latest/notebooks/features/move-prediction.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/move-prediction.ipynb)
- [Run Models on GPU](https://lczerolens.readthedocs.io/en/latest/notebooks/features/run-models-on-gpu.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/run-models-on-gpu.ipynb)
- [Evaluate Models on Puzzles](https://lczerolens.readthedocs.io/en/latest/notebooks/features/evaluate-models-on-puzzles.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/evaluate-models-on-puzzles.ipynb)
- [Convert Official Weights](https://lczerolens.readthedocs.io/en/latest/notebooks/features/convert-official-weights.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/convert-official-weights.ipynb)
- [Visualise Heatmaps](https://lczerolens.readthedocs.io/en/latest/notebooks/features/visualise-heatmaps.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/visualise-heatmaps.ipynb)
- [Probe Concepts](https://lczerolens.readthedocs.io/en/latest/notebooks/features/probe-concepts.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/probe-concepts.ipynb)

### Tutorials

- [Walkthrough](https://lczerolens.readthedocs.io/en/latest/notebooks/walkthrough.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/docs/source/notebooks/walkthrough.ipynb)
- [Framework-Agnostic Interpretability](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/framework-agnostic-interpretability.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/tutorials/framework-agnostic-interpretability.ipynb)
- More to come...

## Demo

### Spaces

Some [Hugging Face Spaces](https://huggingface.co/spaces/lczerolens) are available to try out the library. The demo (:red_circle: in construction) will showcase some of the features of the library and the backends demo makes the conversion of lc0 models to `onnx` easy.

- [Lczerolens Demo](https://huggingface.co/spaces/lczerolens/demo): Showcase some of the features of the library.
- [Lczerolens Backends Demo](https://huggingface.co/spaces/lczerolens/backends-demo): Make the conversion of lc0 models to `onnx` easy.
- [Lczerolens Puzzles Leaderboard](https://huggingface.co/spaces/lczerolens/puzzles-leaderboard): Coming soon...

### Local Demo

Additionally, you can run the gradio demos locally. First you'll need to clone the spaces (after cloning the repo):

```bash
git clone https://huggingface.co/spaces/lczerolens/demo spaces/demo
```

And optionally the backends demo:

```bash
git clone https://huggingface.co/spaces/lczerolens/backends-demo spaces/backends-demo
```

And then launch the demo (running on port `8000`):

```bash
just demo
```

To test the backends use:

```bash
just demo-backends
```

## Full Documentation

See the full [documentation](https://lczerolens.readthedocs.io).

## Contribute

See the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).
