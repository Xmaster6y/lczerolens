# lczerolens quick guide

## What this repo is

`lczerolens` is a PyTorch-first interpretability toolkit for Leela Chess Zero (`lc0`) networks.
It provides:

- loading of lc0 models (`.onnx`, `.pt`, and Hugging Face Hub);
- chess board encoding into lc0-compatible tensors;
- policy/value inference helpers;
- utilities for attribution/probing workflows with `tdhook`, `captum`, `zennit`, and `nnsight`.

## Fast setup

```bash
uv sync
uv run pre-commit install
```

Common project commands:

```bash
just checks
just tests
just docs
```

Optional extras when installing from PyPI:

```bash
pip install "lczerolens[viz]"      # heatmaps / graphviz
pip install "lczerolens[hf]"       # huggingface datasets/hub
pip install "lczerolens[backends]" # lc0 bindings
```

## Code map

- `src/lczerolens/board.py`: `LczeroBoard`, move index mapping, board encoding, heatmap rendering.
- `src/lczerolens/model.py`: `LczeroModel` + flow wrappers (`PolicyFlow`, `WdlFlow`, `ValueFlow`, `ForceValue`).
- `src/lczerolens/data.py`: `GameData`, `BoardData`, `PuzzleData` adapters and dataset helpers.
- `src/lczerolens/sampling.py`: `PolicySampler`, `ModelSampler`, `MCTSSampler`, `SelfPlay`.
- `src/lczerolens/search.py`: MCTS primitives used by `MCTSSampler`.
- `src/lczerolens/constants.py`: 1858-policy move vocabulary (`POLICY_INDEX` and inverse).

## Core mental model

- `LczeroBoard` is the canonical board object; it extends `python-chess` with lc0-specific encoding/decoding.
- `LczeroModel` is a `TensorDictModule` expecting input key `board` and returning TensorDict outputs.
- Input tensor shape is `(112, 8, 8)` per board; policy head shape is `(1858,)` per board.
- Typical output keys: `policy`, `wdl`; optional `value` and `mlh` depending on network heads.
- `ForceValue` adds `value` from `wdl` as `w - l` when a native value head is absent.

## Board encoding details (112 planes)

The default encoder is `InputEncoding.INPUT_CLASSICAL_112_PLANE`.

- Planes `0..103`: 8 history slices x 13 planes each.
  - For each slice: 12 piece planes + 1 repetition plane.
  - Piece ordering is perspective-aware (`us` pieces then `them` pieces).
- Planes `104..107`: castling rights (`us` queen/kingside, then `them` queen/kingside).
- Plane `108`: side-to-move marker (`1` when `us == black`, else `0`).
- Plane `109`: halfmove clock broadcast on all squares.
- Plane `110`: currently unused/reserved (left as zeros).
- Plane `111`: constant ones plane.

Supported input variants:

- `INPUT_CLASSICAL_112_PLANE`: full history when available.
- `INPUT_CLASSICAL_112_PLANE_REPEATED`: repeat earliest available position to fill missing history.
- `INPUT_CLASSICAL_112_PLANE_NO_HISTORY_REPEATED`: repeat current position 8 times.
- `INPUT_CLASSICAL_112_PLANE_NO_HISTORY_ZEROS`: keep only current position, rest of history zeroed.

## Move indexing contract

- Policy logits are indexed over a fixed vocabulary of size `1858` (`constants.py`).
- `LczeroBoard.encode_move(move, us)` maps a legal move to that index.
- `LczeroBoard.decode_move(index)` maps index back to a move in current board context.
- Black-to-move encoding/decoding is normalized through board flipping, so indices stay perspective-consistent.

## Model I/O contract

Load paths:

```python
from lczerolens import LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
# or
model = LczeroModel.from_path("assets/tinygyal-8.onnx")
```

Forward accepts:

- a `LczeroBoard`;
- an iterable of `LczeroBoard`;
- a raw tensor shaped `(112, 8, 8)` or `(B, 112, 8, 8)`;
- a `TensorDict` with key `"board"`.

Returns a TensorDict batch with output keys discovered from the model graph.

## Minimal workflow

```python
from lczerolens import LczeroBoard, LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
board = LczeroBoard()
out = model(board)

best_idx = out["policy"].argmax().item()
best_move = board.decode_move(best_idx)
```

## Interpretability integration pattern

With `tdhook`, keep keys aligned with `LczeroModel`:

- input key: `board`
- output keys: usually `policy`, `wdl` (and optionally `value`, `mlh`)
- nested outputs/attrs use tuple keys, e.g. `("attr", "board")`

Example shape flow:

- `board.to_input_tensor()` -> `(112, 8, 8)`
- `model.prepare_boards(board)` -> `(1, 112, 8, 8)`
- attribution map often -> `(1, 112, 8, 8)` then reduced to `(64,)` or `(8, 8)` for rendering.

## Samplers and search

- `PolicySampler`: choose moves from policy logits over legal indices.
- `ModelSampler`: combines value/WDL, policy, and optional MLH signal into utility.
- `MCTSSampler`: wraps MCTS search with model heuristic for rollout guidance.
- `SelfPlay`: runs white/black samplers into a full game trajectory.

## Common gotchas

- `from_hf` requires `huggingface_hub` (`pip install "lczerolens[hf]"`).
- Always mask policy logits with `board.get_legal_indices()` before interpreting top moves.
- For Integrated Gradients, baseline tensor must match input shape/device exactly.
- Heatmap orientation: start with `heatmap_mode="relative_flip"`.
- `decode_move` defaults to knight promotion when promotion type is ambiguous in policy index.
- `to_input_tensor` temporarily pops move history internally, then restores it; avoid mutating the board concurrently.

## Where to look next

- `skills/lczerolens/SKILL.md`
- `skills/lczerolens/references/api.md`
- `skills/lczerolens/references/tutorials.md`
- https://lczerolens.readthedocs.io
