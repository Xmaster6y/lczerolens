"""Register a dataset in Weights & Biases.

Run with:
```bash
poetry run python -m scripts.register_wandb_dataset
```
"""

import argparse
import os
import random

import wandb
from lczerolens import BoardDataset

#######################################
# HYPERPARAMETERS
#######################################
parser = argparse.ArgumentParser("make-datasets")
parser.add_argument("--output-root", type=str, default=".")
make_dataset = False
seed = 42
train_samples = 10_000
val_samples = 1_000
test_samples = 1_000
log_dataset = False
#######################################

ARGS = parser.parse_args()
os.makedirs(f"{ARGS.output_root}/assets", exist_ok=True)


if make_dataset:
    dataset = BoardDataset("./assets/TCEC_game_collection_random_boards.jsonl")
    all_indices = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(all_indices)
    train_indices = all_indices[:train_samples]
    val_slice = train_samples + val_samples
    val_indices = all_indices[train_samples:val_slice]
    test_slice = val_slice + test_samples
    test_indices = all_indices[val_slice:test_slice]

    dataset.save(
        f"{ARGS.output_root}/assets/"
        "TCEC_game_collection_random_boards_train.jsonl",
        indices=train_indices,
    )
    dataset.save(
        f"{ARGS.output_root}/assets/"
        "TCEC_game_collection_random_boards_val.jsonl",
        indices=val_indices,
    )
    dataset.save(
        f"{ARGS.output_root}/assets/"
        "TCEC_game_collection_random_boards_test.jsonl",
        indices=test_indices,
    )

#  type: ignore
if log_dataset:
    with wandb.init(  # type: ignore
        project="lczerolens-saes", job_type="make-datasets"
    ) as run:
        artifact = wandb.Artifact("tcec_train", type="dataset")  # type: ignore
        artifact.add_file(
            f"{ARGS.output_root}/assets/"
            "TCEC_game_collection_random_boards_train.jsonl"
        )
        run.log_artifact(artifact)
        artifact = wandb.Artifact("tcec_val", type="dataset")  # type: ignore
        artifact.add_file(
            f"{ARGS.output_root}/assets/"
            "TCEC_game_collection_random_boards_val.jsonl"
        )
        run.log_artifact(artifact)
        artifact = wandb.Artifact("tcec_test", type="dataset")  # type: ignore
        artifact.add_file(
            f"{ARGS.output_root}/assets/"
            "TCEC_game_collection_random_boards_test.jsonl"
        )
        run.log_artifact(artifact)
