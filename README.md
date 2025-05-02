<img src="https://raw.githubusercontent.com/Xmaster6y/lczerolens/refs/heads/main/docs/source/_static/images/lczerolens-logo.svg" alt="logo" width="200"/>

# lczerolens 🔍

[![lczerolens](https://img.shields.io/pypi/v/lczerolens?color=purple)](https://pypi.org/project/lczerolens/)
[![license](https://img.shields.io/badge/license-MIT-lightgrey.svg)](https://github.com/Xmaster6y/lczerolens/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![python versions](https://img.shields.io/pypi/pyversions/lczerolens.svg)](https://www.python.org/downloads/)

[![codecov](https://codecov.io/gh/Xmaster6y/lczerolens/graph/badge.svg?token=JKJAWB451A)](https://codecov.io/gh/Xmaster6y/lczerolens)
![ci-fast](https://github.com/Xmaster6y/lczerolens/actions/workflows/ci-fast.yml/badge.svg)
![ci-slow](https://github.com/Xmaster6y/lczerolens/actions/workflows/ci-slow.yml/badge.svg)
![publish](https://github.com/Xmaster6y/lczerolens/actions/workflows/publish.yml/badge.svg)
[![docs](https://readthedocs.org/projects/lczerolens/badge/?version=latest)](https://lczerolens.readthedocs.io/en/latest/?badge=latest)

<a href="https://lczerolens.readthedocs.io"><img src="https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white"></img></a>


Leela Chess Zero (lc0) Lens (`lczerolens`): a set of utilities to make the analysis of Leela Chess Zero networks easy.

## Getting Started

### Installs

```bash
pip install lczerolens
```

### Features

- [Visualise Heatmaps](https://lczerolens.readthedocs.io/en/latest/notebooks/features/visualise-heatmaps.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/visualise-heatmaps.ipynb)
- [Probe Concepts](https://lczerolens.readthedocs.io/en/latest/notebooks/features/probe-concepts.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/probe-concepts.ipynb)
- [Move Prediction](https://lczerolens.readthedocs.io/en/latest/notebooks/features/move-prediction.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/move-prediction.ipynb)
- [Run Models on GPU](https://lczerolens.readthedocs.io/en/latest/notebooks/features/run-models-on-gpu.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/run-models-on-gpu.ipynb)
- [Evaluate Models on Puzzles](https://lczerolens.readthedocs.io/en/latest/notebooks/features/evaluate-models-on-puzzles.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/evaluate-models-on-puzzles.ipynb)
- [Convert Official Weights](https://lczerolens.readthedocs.io/en/latest/notebooks/features/convert-official-weights.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/convert-official-weights.ipynb)

### Tutorials

- [Walkthrough](https://lczerolens.readthedocs.io/en/latest/notebooks/walkthrough.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/docs/source/notebooks/walkthrough.ipynb)
- [Piece Value Estimation Using LRP](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/piece-value-estimation-using-lrp.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/tutorials/piece-value-estimation-using-lrp.ipynb)
- [Evidence of Learned Look-Ahead](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/evidence-of-learned-look-ahead.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/tutorials/evidence-of-learned-look-ahead.ipynb)
- [Train SAEs](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/train-saes.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/tutorials/train-saes.ipynb)

## Demo

### Spaces

- [Lczerolens Demo](https://huggingface.co/spaces/lczerolens/lczerolens-demo)
- [Lczerolens Backends Demo](https://huggingface.co/spaces/lczerolens/lczerolens-backends-demo)
- [Lczerolens Puzzles Leaderboard](https://huggingface.co/spaces/lczerolens/lichess-puzzles-leaderboard)

### Local Demo

Additionally, you can run the gradio demo locally. First you'll need the space, which is packaged in a submodule:

```bash
git clone --recurse-submodules https://github.com/Xmaster6y/lczerolens.git
cd lczerolens
```

And then launch the demo (running on port `8000`):

```bash
make demo
```

To test the backends use:

```bash
make demo-backends
```

## Full Documentation

See the full [documentation](https://lczerolens.readthedocs.io).

## Contribute

See the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).
