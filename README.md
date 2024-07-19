<img src="./docs/source/_static/images/lczerolens-logo.svg" alt="logo" style="width:200px;float:left"/>

# lczerolens :mag:

![ci](https://github.com/Xmaster6y/lczerolens/actions/workflows/ci.yml/badge.svg)
![publish](https://github.com/Xmaster6y/lczerolens/actions/workflows/publish.yml/badge.svg)
<a href="https://pypi.org/project/lczerolens/"><img src="https://img.shields.io/pypi/v/lczerolens?color=purple"></img></a>
[![docs](https://readthedocs.org/projects/lczerolens/badge/?version=latest)](https://lczerolens.readthedocs.io/en/latest/?badge=latest)

<a href="https://lczerolens.readthedocs.io"><img src="https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white"></img></a>


Leela Chess Zero (lc0) Lens (`lczerolens`): a set of utilities to make the analysis of Leela Chess Zero networks easy.

## Getting Started

### Installs

```bash
pip install lczerolens
```

### Features

- [Move Prediction](https://lczerolens.readthedocs.io/en/latest/notebooks/features/move_prediction.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/move_prediction.ipynb)
- [Run Models on GPU](https://lczerolens.readthedocs.io/en/latest/notebooks/features/run_models_on_gpu.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/run_models_on_gpu.ipynb)
- [Evaluate Models on Puzzles](https://lczerolens.readthedocs.io/en/latest/notebooks/features/evaluate_models_on_puzzles.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/evaluate_models_on_puzzles.ipynb)
- [Convert Official Weights](https://lczerolens.readthedocs.io/en/latest/notebooks/features/convert_official_weights.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/features/convert_official_weights.ipynb)

### Tutorials

- [Walkthrough](https://lczerolens.readthedocs.io/en/latest/notebooks/walkthrough.html): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/docs/source/notebooks/walkthrough.ipynb)
- [Piece Value Estimation Using LRP](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/piece-value-estimation-using-lrp.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/tutorials/piece-value-estimation-using-lrp.ipynb)
- [Evidence of Learned Look-Ahead](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/evidence-of-learned-look-ahead.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/tutorials/evidence-of-learned-look-ahead.ipynb)
- [Train SAEs](https://lczerolens.readthedocs.io/en/latest/notebooks/tutorials/train_saes.ipynb): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/lczerolens/blob/main/docs/source/notebooks/tutorials/train_saes.ipynb)

## Demo

:red_circle: Not up to date.

Additionally, you can run the gradio demo locally (also deployed on HF). First you'll need gradio, which is packaged in the `demo` extra:

```bash
pip install lczerolens[demo]
```

And then launch the demo (running on port `8000`):

```bash
make demo
```

## Full Documentation

:red_circle: Not up to date.

See the full [documentation](https://lczerolens.readthedocs.io).

## Contribute

See the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).
