"""Script to generate the base datasets.

Run with:
```bash
poetry run python -m scripts.datasets.make_tcec_dataset
```
"""

import argparse

from datasets import Dataset
from loguru import logger

from scripts.constants import HF_TOKEN


def main(args: argparse.Namespace):
    logger.info(f"Loading `{args.source_file}`...")

    dataset = Dataset.from_json(args.source_file)
    logger.info(f"Loaded dataset: {dataset}")

    if args.push_to_hub:
        logger.info("Pushing to hub...")
        dataset.push_to_hub(
            repo_id=args.dataset_name,
            token=HF_TOKEN,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("make-tcec-dataset")
    parser.add_argument(
        "--source_file",
        type=str,
        default="./assets/tcec-games.jsonl",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lczerolens/tcec-games",
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
