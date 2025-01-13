"""Script to generate the base datasets.

Run with:
```bash
uv run python -m scripts.datasets.make_lichess_dataset
```
"""

import argparse

from datasets import Dataset
from loguru import logger

from scripts.constants import HF_TOKEN


def main(args: argparse.Namespace):
    logger.info(f"Loading `{args.source_file}`...")

    dataset = Dataset.from_csv(args.source_file)
    logger.info(f"Loaded dataset: {dataset}")

    if args.push_to_hub:
        logger.info("Pushing to hub...")
        dataset.push_to_hub(
            repo_id=args.dataset_name,
            token=HF_TOKEN,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("make-lichess-dataset")
    parser.add_argument(
        "--source_file",
        type=str,
        default="./assets/lichess_db_puzzle.csv",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lczerolens/lichess-puzzles",
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
