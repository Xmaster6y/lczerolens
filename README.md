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

### Load a Model

Load a leela network from file (already converted to `onnx`):

```python
from lczerolens import LczeroModel

model = LczeroModel.from_path(
    "leela-network.onnx"
)
```

To convert original weights see the section [Convert Official Weights](#convert-official-weights).

### Predict a Move

The defined model natively integrates with `python-cess`. Use the utils to predict a policy vector and obtain an UCI move:

```python
import chess

from lczerolens.encodings import move as move_utils

# Get the model predictions
board = chess.Board()
out = model(board)
policy = out["policy"]

# Use the prediction
best_move_index = policy.argmax()
move = move_utils.decode_move(best_move_index)
board.push(move)

print(uci_move, board)
```

As most network are trained only on legal moves the argmax should only be computed on them:

```
legal_indices = move_utils.get_legal_indices(board)
legal_policy = policy[0].gather(0, legal_indices)
best_move_index = legal_indices[legal_policy.argmax()]
move = move_utils.decode_move(best_move_index)
```

As this becomes cumbersome you should turn to the `Sampler` class, which now reads:


```python
from lczerolens.play import PolicySampler

sampler = PolicySampler(model, use_argmax=True)
move = sampler.get_next_move(board)
```

### Compute a Heatmap

:red_circle: Not up to date.

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

### Convert Official Weights

To convert a network you'll need to have installed the `lc0` extra:

```bash
pip install lczerolens[lc0]
```

You can convert networks to `onnx` using the official `lc0` binaries or
by using the `backends` module:

```python
from lczerolens.encodings import backends

backends.convert_to_onnx(
    "leela-network.pb.gz",
    "leela-network.onnx"
)
```

Only the latest networks are supported, in order to build older weights you should build the associated binaries.

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

:red_circle: Documentation coming soon.

## Contribute

See the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md).
