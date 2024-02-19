<img src="./docs/source/_static/images/lczerolens-logo.svg" alt="logo" style="width:200px;float:left"/>

# lczerolens :mag:

![ci](https://github.com/Xmaster6y/lczerolens/actions/workflows/ci.yml/badge.svg)
![publish](https://github.com/Xmaster6y/lczerolens/actions/workflows/publish.yml/badge.svg)
<a href="https://pypi.org/project/lczerolens/"><img src="https://img.shields.io/pypi/v/lczerolens?color=purple"></img></a>
[![docs](https://readthedocs.org/projects/lczerolens/badge/?version=latest)](https://lczerolens.readthedocs.io/en/latest/?badge=latest)

<a href="https://lczerolens.readthedocs.io"><img src="https://img.shields.io/badge/-Read%20the%20Docs%20Here-blue?style=for-the-badge&logo=Read-the-Docs&logoColor=white"></img></a>


Leela Chess Zero (lc0) Lens: a set of utilities to make the analysis of Leela Chess Zero networks easy.

## Getting Started

### Installs

```bash
pip install lczerolens
```

### Load a Model

Build a model from leela file (convert then load):

```python
from lczerolens import AutoBuilder

model = AutoBuilder.from_path(
    "leela-network.onnx"
)
```

### Predict a Move

Use a wrapper and the utils to predict a policy vector and obtain an UCI move:

```python
import chess
import torch

from lczerolens import move_utils, ModelWrapper

# Wrap the model
wrapper = ModelWrapper(model)
board = chess.Board()

# Get the model predictions
out = wrapper.predict(board)
policy = out["policy"]

# Use the prediction
best_move_index = policy.argmax()
move = move_utils.decode_move(best_move_index)
board.push(move)

print(uci_move, board)
```

### Compute a Heatmap

Use a lens to compute a heatmap

```python
from lczerolens import visualisation_utils, LrpLens

# Get the lens
lens = LrpLens()

# Compute the relevance
assert lens.is_compatible(wrapper)
relevance = lens.compute_heatmap(board, wrapper)

# Choose a plane index and render the heatmap on the board
plane_index = 0
heatmap = relevance_tensor[plane_index].view(64)
if board.turn == chess.BLACK:
    heatmap = heatmap.view(8, 8).flip(0).view(64)
svg_board, fig = visualisation_utils.render_heatmap(
    board, heatmap, normalise="abs"
)
```

### Convert a Network

To convert a network you'll need to have installed the `lc0` extra:

```bash
pip install lczerolens[lc0]
```

```python
from lczerolens import lczero as lczero_utils

lczero_utils.convert_to_onnx(
    "leela-network.pb.gz",
    "leela-network.onnx"
)
```

## Demo

Additionally, you can run the gradio demo locally (also deployed on HF). First you'll need gradio, which is packaged in the `demo` extra:

```bash
pip install lczerolens[demo]
```

And then launch the demo (running on port `8000`):

```bash
make demo
```

## Full Documentation

:red_circle: Documentation coming soon.

## Contribute

See the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).
