---
name: lczerolens
description: Provides guidance for interpretability of Leela Chess Zero (lc0) neural networks. Use when loading lc0 models, encoding chess boards, running inference, visualizing heatmaps, probing concepts, evaluating on puzzles, or pairing with tdhook, captum, zennit, or nnsight for XAI.
version: 1.0.0
author: Xmaster6y
license: MIT
tags: [lczerolens, lc0, Chess, Interpretability, Leela Chess Zero, TensorDict, PyTorch, Heatmaps, Probing, Concepts]
dependencies: [lczerolens, tensordict>=0.9.1, torch>=2.5.0, python-chess>=1.999, huggingface-hub>=0.27.1]
---

# lczerolens

Interpretability for Leela Chess Zero (lc0) networks. Framework-agnostic (PyTorch): pair with `tdhook`, `captum`, `zennit`, or `nnsight`.

**Docs**: [lczerolens.readthedocs.io](https://lczerolens.readthedocs.io) · **GitHub**: [Xmaster6y/lczerolens](https://github.com/Xmaster6y/lczerolens)

## When to Use

**Use lczerolens when you need to:**
- Load and run lc0 models (ONNX, PyTorch, Hugging Face Hub)
- Encode chess boards into 112×8×8 input tensors
- Run interpretability methods (attribution, probing, LRP) on chess NNs
- Visualize saliency heatmaps on the board
- Probe concepts (material, threats, best move) on board representations
- Evaluate models on puzzles or games
- Use MCTS, policy/value sampling, or self-play

**Consider alternatives:** tdhook (attribution, probing, steering), nnsight (remote 70B+), general chess engines (Stockfish) for non-NN analysis.

---

## Workflow 1: Load Model and Run Inference

**Goal**: Get policy and value predictions for a chess position.

**Checklist**:
- [ ] Load model: `LczeroModel.from_hf(repo_id)` or `from_path(path)`
- [ ] Create board: `LczeroBoard()` or `LczeroBoard(fen)`
- [ ] Call `model(board)` → TensorDict with `policy`, `wdl`/`value`, optionally `mlh`
- [ ] Use `board.decode_move(idx)` for move index → UCI

```python
from lczerolens import LczeroBoard, LczeroModel

model = LczeroModel.from_hf("lczerolens/maia-1100")
board = LczeroBoard()

output = model(board)
best_move_idx = output["policy"].argmax().item()
print(board.decode_move(best_move_idx))
```

---

## Workflow 2: Encode Boards and Prepare Inputs

**Goal**: Convert chess positions to model input tensors.

**Checklist**:
- [ ] Use `board.to_input_tensor(input_encoding=...)` for 112×8×8 tensor
- [ ] Default: `InputEncoding.INPUT_CLASSICAL_112_PLANE` (8 history planes + castling, color, etc.)
- [ ] For model: `model.prepare_boards(board1, board2, ...)` batches boards
- [ ] Move encoding: `LczeroBoard.encode_move(move, us)` and `board.decode_move(index)`

```python
from lczerolens import LczeroBoard
from lczerolens.board import InputEncoding

board = LczeroBoard("r1bqkbnr/pppppppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")
tensor = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE)
# tensor shape: (112, 8, 8)
```

---

## Workflow 3: Framework-Agnostic Interpretability (tdhook)

**Goal**: Run attribution, probing, or patching with tdhook on lc0 models.

**Checklist**:
- [ ] lczerolens uses `in_keys=["board"]`, `out_keys=["policy","wdl",...]`
- [ ] Build TensorDict: `TensorDict({"board": x}, batch_size=...)`
- [ ] For tdhook: pass `in_keys=["board"]`, `out_keys=["policy","wdl"]` to `prepare()`
- [ ] Attribution output: `td.get(("attr", "board"))` → 112×8×8 or 64 for heatmap
- [ ] Use `board.render_heatmap(heatmap)` for visualization (requires `lczerolens[viz]`)

```python
from lczerolens import LczeroBoard, LczeroModel
from tdhook.attribution import IntegratedGradients
from tensordict import TensorDict

model = LczeroModel.from_hf("lczerolens/maia-1100")
board = LczeroBoard()

def init_attr_targets(td, _):
    idx = td["policy"].argmax(dim=-1)
    return TensorDict(out=td["policy"].gather(-1, idx.unsqueeze(-1)).squeeze(-1), batch_size=td.batch_size)

baseline = board.to_input_tensor()
baseline[:] = 0  # or neutral baseline
x = model.prepare_boards(board)
td = TensorDict({"board": x, ("baseline", "board"): baseline.unsqueeze(0)}, batch_size=1)

with IntegratedGradients(init_attr_targets=init_attr_targets).prepare(
    model, in_keys=["board"], out_keys=["policy", "wdl"]
) as hooked:
    td = hooked(td)
    attr = td.get(("attr", "board"))  # (1, 112, 8, 8) or aggregate to 64
```

---

## Workflow 4: Probe Concepts on Board Representations

**Goal**: Train probes on representations for classification (e.g. material advantage, threats).

**Checklist**:
- [ ] Define `Concept` (e.g. `HasMaterialAdvantage`, `HasThreat`, `HasPiece`)
- [ ] Use `GameData.to_boards(concept=concept)` or `BoardData` with `concept_collate_fn`
- [ ] For tdhook Probing: `additional_keys=["labels","step_type"]`, `step_type="fit"` / `"predict"`
- [ ] lc0 model structure: `block0`, `block1`, ... (residual blocks); use regex for layer selection

```python
from lczerolens import LczeroBoard, LczeroModel
from lczerolens.concepts import HasMaterialAdvantage, HasPiece
from lczerolens.data import GameData, BoardData

concept = HasMaterialAdvantage(relative=True)
# Or: OrBinaryConcept(HasPiece("Q"), HasPiece("R"))

game = GameData.from_dict({"gameid": "1", "moves": "e4 e5 Nf3 Nc6 ..."})
boards = game.to_boards(n_history=7, concept=concept, output_dict=True)
# Each dict has "fen", "moves", "label"
```

---

## Workflow 5: Visualize Heatmaps

**Goal**: Render attribution or policy heatmaps on the chess board.

**Checklist**:
- [ ] Install `lczerolens[viz]` (matplotlib, graphviz)
- [ ] Heatmap: `(64,)` or `(8,8)`; use `heatmap_mode`: `"relative_flip"` (default), `"relative_rotation"`, `"absolute"`
- [ ] `board.render_heatmap(heatmap, save_to="out.svg", ...)` returns SVG or saves files

```python
from lczerolens import LczeroBoard
import torch

board = LczeroBoard()
# heatmap: (64,) or (8,8) from attribution
heatmap = torch.randn(64)  # example
svg_board, svg_colorbar = board.render_heatmap(heatmap, normalise="abs")
# Or: board.render_heatmap(heatmap, save_to="out.svg")
```

---

## Workflow 6: Evaluate on Puzzles

**Goal**: Score a model on Lichess puzzles.

**Checklist**:
- [ ] Use `PuzzleData` from dataset; `puzzle.initial_board` for position after first move
- [ ] Create `ModelSampler` or `PolicySampler` with model
- [ ] `puzzle.evaluate(sampler)` or `PuzzleData.evaluate_multiple(puzzles, sampler)`
- [ ] Returns `(score, perplexity)` or `{"score", "perplexity", "normalized_perplexity"}`

```python
from lczerolens import LczeroBoard, LczeroModel
from lczerolens.data import PuzzleData
from lczerolens.sampling import PolicySampler

model = LczeroModel.from_hf("lczerolens/maia-1100")
sampler = PolicySampler(model=model, use_argmax=True)

puzzle = PuzzleData.from_dict({...})  # from datasets
score, perplexity = puzzle.evaluate(sampler)
```

---

## Key API Reference

| Class / Method | Purpose |
|----------------|---------|
| `LczeroModel.from_hf(repo_id)` | Load from Hugging Face Hub |
| `LczeroModel.from_path(path)` | Load .onnx or .pt |
| `LczeroBoard(fen)` | Create board from FEN |
| `board.to_input_tensor()` | 112×8×8 input tensor |
| `board.encode_move(move, us)` | Move → policy index |
| `board.decode_move(index)` | Policy index → Move |
| `board.render_heatmap(heatmap)` | Visualize on board |
| `model.prepare_boards(*boards)` | Batch boards for forward |
| `PolicyFlow`, `ValueFlow`, `WdlFlow` | Isolate policy/value/wdl heads |
| `GameData`, `BoardData`, `PuzzleData` | Data for games/boards/puzzles |
| `ModelSampler`, `PolicySampler`, `MCTSSampler` | Move selection |
| `Concept`, `HasMaterialAdvantage`, `HasThreat` | Concept labels for probing |

---

## Model Output Keys

| Key | Shape | Description |
|-----|-------|-------------|
| `board` | (B, 112, 8, 8) | Input (from TensorDict) |
| `policy` | (B, 1858) | Move logits |
| `wdl` | (B, 3) | Win-Draw-Loss probs |
| `value` | (B,) | Native value head (if present); use `ForceValue` to derive from wdl when absent |
| `mlh` | (B,) | Moves-left head (optional) |

---

## Common Issues & Troubleshooting

| Issue | Solution |
|-------|----------|
| `KeyError` with tdhook | Use `in_keys=["board"]`, `out_keys=["policy","wdl"]` for LczeroModel |
| Heatmap wrong orientation | Use `heatmap_mode="relative_flip"` (default) for side-to-move view |
| `ImportError` for viz | `pip install lczerolens[viz]` |
| Hugging Face load fails | `pip install lczerolens[hf]` |
| Policy index out of range | Use `board.get_legal_indices()` to mask; policy has 1858 indices |
| ONNX load error | Ensure `onnx2torch` compatible; try `from_torch_path` if converted |

See [references/api.md](references/api.md) for more details.

---

## Setup & Installation

```bash
pip install lczerolens
```

Optional extras:
```bash
pip install lczerolens[viz]    # heatmaps, graphviz
pip install lczerolens[hf]     # Hugging Face Hub
pip install lczerolens[backends]  # lc0 bindings for conversion
```

---

## Integration with Interpretability Frameworks

| Framework | Use Case | Key Adapter |
|-----------|----------|-------------|
| **tdhook** | Attribution, probing, steering, patching | `in_keys=["board"]`, `out_keys` from model |
| **captum** | IntegratedGradients, Saliency | Wrap `model.module` or use TensorDict → tensor |
| **zennit** | LRP, composite rules | Same as captum; board-aligned output |
| **nnsight** | Activation analysis, remote | `LanguageModel` not applicable; use trace on `model.module` |

---

## References

| File | Contents |
|------|----------|
| [references/README.md](references/README.md) | Overview |
| [references/api.md](references/api.md) | Full API: LczeroModel, LczeroBoard, data, concepts |
| [references/tutorials.md](references/tutorials.md) | Notebooks and use-case tutorials |

**Official docs**: [lczerolens.readthedocs.io](https://lczerolens.readthedocs.io)
