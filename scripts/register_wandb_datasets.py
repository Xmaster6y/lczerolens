"""Register datasets in Weights & Biases.

Run with:
```bash
poetry run python -m scripts.register_wandb_datasets \
    --make_datasets --log_datasets
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
parser = argparse.ArgumentParser("register-wandb-datasets")
parser.add_argument("--output_root", type=str, default=".")
parser.add_argument(
    "--make_datasets", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--train_samples", type=int, default=100_000)
parser.add_argument("--val_samples", type=int, default=5_000)
parser.add_argument("--test_samples", type=int, default=5_000)
parser.add_argument(
    "--log_datasets", action=argparse.BooleanOptionalAction, default=False
)
#######################################

ARGS = parser.parse_args()
os.makedirs(f"{ARGS.output_root}/assets", exist_ok=True)

if ARGS.make_datasets:
    dataset = BoardDataset("./assets/TCEC_game_collection_random_boards.jsonl")
    all_indices = list(range(len(dataset)))
    random.seed(ARGS.seed)
    random.shuffle(all_indices)
    train_indices = all_indices[: ARGS.train_samples]
    val_slice = ARGS.train_samples + ARGS.val_samples
    val_indices = all_indices[ARGS.train_samples : val_slice]
    test_slice = val_slice + ARGS.test_samples
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

if ARGS.log_datasets:
    wandb.login()  # type: ignore
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
