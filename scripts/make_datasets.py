"""Script to generate datasets.

Run with:
```bash
poetry run python -m scripts.make_datasets
```
"""

from lczerolens import BoardDataset, GameDataset

#######################################
# HYPERPARAMETERS
#######################################
dataset_name = "test_stockfish_5000.jsonl"
#######################################


dataset = GameDataset(f"./assets/{dataset_name}")
board_dataset = BoardDataset.from_game_dataset(dataset, n_history=7)
print(f"[INFO] Board dataset len: {len(board_dataset)}")
board_dataset.save(
    f"./assets/{dataset_name.replace('.jsonl', '_boards.jsonl')}"
)
