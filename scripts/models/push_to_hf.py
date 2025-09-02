"""Utility script to upload a (single-file) model artefact to the 🤗 Hub.

Example
-------
uv run python -m scripts.models.push_to_hf \
    --model_path ./assets/maia-1100.onnx \
    --repo_id lczerolens/maia-1100 \
    --push_to_hub
"""

import argparse

from loguru import logger

from scripts.constants import HF_TOKEN
from lczerolens.model import LczeroModel


def main(args: argparse.Namespace) -> None:
    logger.info("Loading model…")
    model = LczeroModel.from_path(args.model_path)

    if args.push_to_hub:
        model.push_to_hf(
            repo_id=args.repo_id, create_if_not_exists=True, create_kwargs={"token": HF_TOKEN}, token=HF_TOKEN
        )
    else:
        logger.warning("--push_to_hub was not supplied, skipping actual upload. Dry-run only.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("push-model-to-hf")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model file to upload (e.g. a .onnx, .bin, .pt, …)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Destination repository on the Hub in the form <user>/<repo> or <org>/<repo>.",
    )
    parser.add_argument(
        "--push_to_hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually push to the Hub. When omitted, perform a dry-run only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
