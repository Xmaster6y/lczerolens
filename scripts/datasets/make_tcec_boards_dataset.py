"""Script to convert a dataset of games into a dataset of boards.

Run with:
```bash
uv run python -m scripts.datasets.make_tcec_boards_dataset --push_to_hub
```
"""

import argparse

from datasets import load_dataset
from loguru import logger

from lczerolens.data import GameData, BoardData, columns_to_rows, rows_to_columns
from scripts.constants import HF_TOKEN


def main(args: argparse.Namespace):
    logger.info(f"Loading `{args.source_dataset}`...")
    dataset = load_dataset(args.source_dataset, split="train")
    logger.info(f"Loaded dataset with {len(dataset)} games")

    def game_to_boards(columns):
        all_boards = []
        rows = columns_to_rows(columns)
        for row in rows:
            game = GameData.from_dict(row)
            all_boards.extend(
                game.to_boards(
                    n_history=args.n_history,
                    skip_book_exit=args.skip_book_exit,
                    skip_first_n=args.skip_first_n,
                    output_dict=True,
                    concept=None,
                )
            )
        return rows_to_columns(all_boards)

    boards_dataset = dataset.map(
        game_to_boards,
        batched=True,
        batch_size=10,
        remove_columns=dataset.column_names,
        features=BoardData.get_dataset_features(),
        num_proc=4,
    ).flatten()

    logger.info(f"Constructed {len(boards_dataset)} boards")
    logger.info(f"Created boards dataset: {boards_dataset}")

    if args.push_to_hub:
        logger.info("Pushing to hub...")
        boards_dataset.push_to_hub(
            repo_id=args.dataset_name,
            token=HF_TOKEN,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("make-tcec-boards-dataset")
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="lczerolens/tcec-games",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="lczerolens/tcec-boards",
    )
    parser.add_argument(
        "--n_history",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--skip_book_exit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip_first_n",
        type=int,
        default=0,
    )
    parser.add_argument("--push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
