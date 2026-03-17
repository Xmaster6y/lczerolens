# lczerolens API Reference

Key classes, usage patterns, and method implementations.

## LczeroModel

TensorDictModule wrapper for lc0 networks. Input key: `board`, output keys: `policy`, `wdl`/`value`, optionally `mlh`.

```python
# Load
model = LczeroModel.from_hf("lczerolens/maia-1100")
model = LczeroModel.from_path("model.onnx")  # or .pt
model = LczeroModel.from_onnx_path("model.onnx")
model = LczeroModel.from_torch_path("model.pt")

# Forward
output = model(board)  # LczeroBoard or (board1, board2)
output = model(TensorDict({"board": x}, batch_size=B))
output = model(tensor)  # (B, 112, 8, 8) auto-wrapped

# Prepare boards for batching
x = model.prepare_boards(board1, board2, input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE)
```

### Flow Variants

- `PolicyFlow.from_model(model)`: Isolate policy head only
- `ValueFlow.from_model(model)`: Isolate value head
- `WdlFlow.from_model(model)`: Isolate WDL head
- `ForceValue.from_model(model)`: Force value flow (for MCTS heuristic)

## LczeroBoard

Subclass of `chess.Board` with lc0-specific encoding.

```python
board = LczeroBoard()
board = LczeroBoard("r1bqkbnr/pppppppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1")

# Input tensor
tensor = board.to_input_tensor(input_encoding=InputEncoding.INPUT_CLASSICAL_112_PLANE)
# Shape: (112, 8, 8)

# Move encoding
idx = LczeroBoard.encode_move(move, board.turn)
move = board.decode_move(idx)

# Legal indices
legal = board.get_legal_indices()  # shape (n_legal,)

# Next boards
for next_board in board.get_next_legal_boards(n_history=7):
    ...

# Heatmap
board.render_heatmap(heatmap, normalise="abs", save_to="out.svg")
```

### InputEncoding

- `INPUT_CLASSICAL_112_PLANE`: 8 history planes (13 each) + castling + color + halfmove + etc.
- `INPUT_CLASSICAL_112_PLANE_REPEATED`: Same, repeat last position for empty history
- `INPUT_CLASSICAL_112_PLANE_NO_HISTORY_REPEATED`: No history, repeat current
- `INPUT_CLASSICAL_112_PLANE_NO_HISTORY_ZEROS`: No history, zeros

## Data Classes

### GameData

```python
game = GameData.from_dict({"gameid": "1", "moves": "e4 e5 Nf3 ..."})
boards = game.to_boards(n_history=7, concept=concept, output_dict=True)
```

### BoardData

```python
# From dataset row
board, labels = BoardData.concept_collate_fn(batch, concept=concept)
```

### PuzzleData

```python
puzzle = PuzzleData.from_dict({...})
board = puzzle.initial_board  # after first move
for b, m in puzzle.board_move_generator(all_moves=False):
    ...
metrics = puzzle.evaluate(sampler)
```

## Concepts

```python
# Binary
HasPiece("Q", relative=True)
HasMaterialAdvantage(relative=True)
HasThreat("Q", relative=True)
HasMateThreat()
OrBinaryConcept(c1, c2)
AndBinaryConcept(c1, c2)

# Multiclass
BestLegalMove(model)

# For probing: concept.compute_label(board), concept.get_dataset_feature()
```

## Samplers

```python
ModelSampler(model=model, use_argmax=True)  # alpha*value + beta*mlh + gamma*policy
PolicySampler(model=model, use_argmax=True)  # policy only
MCTSSampler(model=model, num_simulations=100)
RandomSampler()

# Usage
for board in boards:
    utility, legal_indices, to_log = next(sampler.get_utilities(iter([board])))
    move = sampler.choose_move(board, utility, legal_indices)
```

## SelfPlay

```python
from lczerolens.sampling import SelfPlay, PolicySampler

sp = SelfPlay(white=PolicySampler(model), black=PolicySampler(model))
moves, final_board = sp.play(max_moves=100)
```

## Module Path Resolution (for tdhook)

lc0 models typically have structure:
- `inputconv`, `block0`, `block1`, ... (residual blocks with `conv1`, `conv2`, SE blocks)
- Policy/value heads at end

Use regex for layer selection, e.g. `"block[0-9]+\\.conv2$"` for block outputs.
