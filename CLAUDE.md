# lczerolens quick guide

## What this repo is

`lczerolens` makes lc0-family models portable and operable in PyTorch, then
expresses their evaluator and search behavior as chess-domain evidence. It owns
the lc0 interoperability and chess-analysis boundary; neural-interpretability
methods remain external integrations.

It provides:

- loading of lc0 models (`.onnx`, `.pt`, and Hugging Face Hub);
- chess board encoding into lc0-compatible tensors;
- a stable evaluator contract for policy, WDL, value, and MLH outputs;
- typed, capability-aware search traces and a deterministic reference MCTS for
  decision evidence.

## Fast setup

```bash
uv sync --group dev
uv run --group dev pre-commit install
```

The docs command needs its own dependency group:

```bash
uv sync --group docs
```

Common project commands:

```bash
just checks
just tests
just docs
```

Optional extras when installing from PyPI:

```bash
pip install "lczerolens[hub]"      # Hugging Face Hub loading/publishing
pip install "lczerolens[backends]" # lc0 bindings
```

## Code map

- `src/lczerolens/_codec/`: stateless Lczero input-plane and policy-vocabulary transport.
- `src/lczerolens/evaluator.py`: chess-aware TensorDict preparation and standardized evaluation.
- `src/lczerolens/model.py`: `LczeroModel` + flow wrappers (`PolicyFlow`, `WdlFlow`, `ValueFlow`, `ForceValue`).
- `src/lczerolens/search/`: unified typed limits and results, engine-independent
  traces, deterministic reference search, official Lczero adapter, and replay helpers.
- `src/lczerolens/constants.py`: 1858-policy move vocabulary (`POLICY_INDEX` and inverse).

## Core mental model

- `chess.Board` is the canonical position and history object; lczerolens does not subclass it.
- `LczeroModel` owns loading and raw TensorDict execution; `LczeroEvaluator` owns chess semantics.
- Input tensor shape is `(112, 8, 8)` per board; policy head shape is `(1858,)` per board.
- Typical output keys: `policy`, `wdl`; optional `value` and `mlh` depending on network heads.
- `ForceValue` adds `value` from `wdl` as `w - l` when a native value head is absent.

## Board encoding details (112 planes)

The evaluator defaults to `InputFormat.CLASSICAL_112`.

- Planes `0..103`: 8 history slices x 13 planes each.
  - For each slice: 12 piece planes + 1 repetition plane.
  - Piece ordering is perspective-aware (`us` pieces then `them` pieces).
- Planes `104..107`: castling rights (`us` queen/kingside, then `them` queen/kingside).
- Plane `108`: side-to-move marker (`1` when `us == black`, else `0`).
- Plane `109`: halfmove clock broadcast on all squares.
- Plane `110`: currently unused/reserved (left as zeros).
- Plane `111`: constant ones plane.

Supported input variants:

- `CLASSICAL_112`: full history when available.
- `CLASSICAL_112_REPEATED`: repeat earliest available position to fill missing history.
- `NO_HISTORY_REPEATED`: repeat current position 8 times.
- `NO_HISTORY_ZEROS`: keep only the current position; older history planes are zero.

## Move indexing contract

- Policy logits are indexed over a fixed vocabulary of size `1858` (`constants.py`).
- The private stateless codec maps moves and indices in a board context.
- Users consume legal moves through `Evaluation.policy` rather than raw indices.
- Black-to-move encoding/decoding is normalized through board flipping, so indices stay perspective-consistent.

## Model I/O contract

Load paths:

```python
from lczerolens import LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
# or
model = LczeroModel.from_path("assets/tinygyal-8.onnx")
```

``LczeroModel`` executes a TensorDict containing its declared input key and
returns a TensorDict with heads discovered from the model graph. Use
``LczeroEvaluator`` for board encoding, legal policy semantics, and batching.

## Minimal workflow

```python
import chess
from lczerolens import LczeroEvaluator, LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
evaluator = LczeroEvaluator(model)
board = chess.Board()
evaluation = evaluator.evaluate(board)
best_move = evaluation.policy.best_move
```

## Architecture boundaries

- `python-chess` owns chess rules, FENs, and legal moves.
- `lczerolens` owns lc0 encoding, move-vocabulary transport, the PyTorch
  evaluator contract, and chess-decision evidence.
- External packages such as `tdhook`, `captum`, `zennit`, and `nnsight` own
  attribution, probing, hooks, patches, and other neural-method semantics.
- `ReferenceSearch` is deterministic reference search for evaluation and replay;
  `LczeroSearch` translates only evidence exposed by the official engine.

Search consumers must call `SearchTrace.supports()` or `SearchTrace.require()`
before making capability-dependent claims. A trace records evidence at the
capability level it actually provides.

## External interpretability integration pattern

With `tdhook`, instrument `LczeroEvaluator.model` using the canonical keys:

- input key: `("input", "planes")`
- network output keys: `("network", "policy_logits")`, `("network", "wdl")`,
  and optional `value` or `mlh` counterparts
- nested outputs and instrumentation keys remain in the same TensorDict

Example shape flow:

- `evaluator.prepare([board])[("input", "planes")]` -> `(1, 112, 8, 8)`
- attribution map often -> `(1, 112, 8, 8)` before a downstream method reduces it.

## Common gotchas

- `from_hf` requires `huggingface_hub` (`pip install "lczerolens[hub]"`).
- Use `Evaluation.policy` before interpreting move preferences; it is legal-move aware.
- For Integrated Gradients, baseline tensor must match input shape/device exactly.
- The stateless encoder copies history and never mutates the caller's board.

## Where to look next

- https://lczerolens.readthedocs.io
