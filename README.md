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

- [Move Prediction](move_prediction): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/docs/source/notebooks/features/move-prediction.ipynb)
- [Convert Official Weights](convert_official_weights): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/docs/source/notebooks/features/convert-official-weights.ipynb)

### Tutorials

- [Walkthrough](walkthrough): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/docs/source/notebooks/walkthrough.ipynb)
- [Piece Value Estimation Using LRP](piece-value-estimation-using-lrp): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/docs/source/notebooks/tutorials/piece-value-estimation-using-lrp.ipynb)
- [Evidence of Learned Look-Ahead](evidence-of-learned-look-ahead): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Xmaster6y/docs/source/notebooks/tutorials/evidence-of-learned-look-ahead.ipynb)

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
