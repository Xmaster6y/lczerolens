"""Script to filter a Lichess puzzle dataset by mate in N moves.

Run with:
```bash
uv run python -m scripts.datasets.make_lichess_matein_dataset --mate_in 3 --push_to_hub
```
"""

from typing import Optional
import argparse
import math
import chess
from datasets import load_dataset
from loguru import logger
from scripts.constants import HF_TOKEN


def compute_mate_length(fen: str, moves_str: str) -> int | None:
    """Simulate the puzzle from FEN and compute mate length (in full moves), ignoring the first move of the puzzle."""
    board = chess.Board(fen)
    moves = moves_str.split()

    for ply_index, move_uci in enumerate(moves, start=1):
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None  # illegal move
        board.push(move)
        if board.is_checkmate():
            # Exclude the first ply (first move)
            return math.ceil((ply_index - 1) / 2)

    return None  # No mate found


def main(args: argparse.Namespace):
    logger.info(f"Loading dataset `{args.source_dataset}`...")
    dataset = load_dataset(args.source_dataset, split="train")
    logger.info(f"Loaded dataset with {len(dataset)} entries")

    dataset = dataset.filter(lambda row: compute_mate_length(row["FEN"], row["Moves"]) == args.mate_in)
    logger.info(f"Filtered dataset to mate in {args.mate_in} => {len(dataset)} entries")

    target_name = args.dataset_name or f"{args.source_dataset}-mate-in-{args.mate_in}"

    if args.push_to_hub:
        logger.info(f"Pushing dataset `{target_name}` to Hugging Face Hub...")
        dataset.push_to_hub(repo_id=target_name, token=HF_TOKEN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("filter-lichess-dataset-by-mate")

    parser.add_argument(
        "--source_dataset",
        type=str,
        default="lczerolens/lichess-puzzles",
    )

    parser.add_argument(
        "--dataset_name",
        type=Optional[str],
        default=None,
    )

    parser.add_argument(
        "--push_to_hub",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--mate_in", type=int, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
