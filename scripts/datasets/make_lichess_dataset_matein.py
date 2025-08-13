"""
Script to filter a Lichess puzzle dataset by mate in N moves using python-chess.

It loads a dataset (local or Hugging Face), simulates the puzzle moves, and filters
puzzles that are checkmate in the given number of moves (mate in N).

Example usage:
uv run python -m scripts.datasets.make_lichess_dataset_matein --source_dataset lczerolens/lichess-puzzles --mate 3
"""

from typing import Optional
import argparse
import math
import chess
from datasets import load_dataset
from loguru import logger
from scripts.constants import HF_TOKEN


def compute_mate_length(fen: str, moves_str: str) -> int | None:
    """
    Simulate the puzzle from FEN and compute mate length (in full moves),
    ignoring the first move of the puzzle.

    Args:
        fen: starting board position in FEN
        moves_str: space-separated UCI moves

    Returns:
        Mate length in full moves (int) or None if no mate
    """
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
    # Load dataset (local CSV or Hugging Face dataset)
    logger.info(f"Loading dataset `{args.source_dataset}`...")
    dataset = load_dataset(args.source_dataset, split="train")
    logger.info(f"Loaded dataset with {len(dataset)} entries")

    # Filter by requested mate length
    dataset = dataset.filter(lambda row: compute_mate_length(row["FEN"], row["Moves"]) == args.mate)
    logger.info(f"Filtered dataset to mate in {args.mate} => {len(dataset)} entries")

    # Determine target dataset name
    target_name = args.target_dataset or f"{args.source_dataset}-M{args.mate}"

    # Push to Hub or save locally
    if args.push_to_hub:
        logger.info(f"Pushing dataset `{target_name}` to Hugging Face Hub...")
        dataset.push_to_hub(repo_id=target_name, token=HF_TOKEN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("filter-lichess-dataset-by-mate")

    parser.add_argument(
        "--source_dataset",
        type=str,
        default="lczerolens/lichess-puzzles",
        help="Dataset path or Hugging Face dataset ID to load.",
    )

    parser.add_argument(
        "--target_dataset",
        type=Optional[str],
        default=None,
        help="Name of the output dataset. If None, source_dataset + suffix is used.",
    )

    parser.add_argument(
        "--push_to_hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to push the filtered dataset to Hugging Face Hub.",
    )

    parser.add_argument("--mate", type=int, required=True, help="Number of moves until mate (first move ignored).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
