"""Utility script to upload an existing file to the 🤗 Hub.

Example
-------
uv run python -m scripts.models.upload_file \
  --model_path ./assets/maia-1100.onnx  \
  --path_in_repo model.onnx \
  --repo_id lczerolens/maia-1100 \
  --push_to_hub
"""

import argparse
from pathlib import Path

from huggingface_hub import create_repo, upload_file
from loguru import logger

from scripts.constants import HF_TOKEN


def main(args: argparse.Namespace) -> None:
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"File not found: {model_path}")

    path_in_repo = args.path_in_repo or model_path.name

    if args.push_to_hub:
        logger.info(f"Uploading `{model_path}` to `{args.repo_id}:{path_in_repo}`…")

        # Ensure repo exists (idempotent)
        create_repo(args.repo_id, token=HF_TOKEN, exist_ok=True, repo_type="model")

        upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            token=HF_TOKEN,
        )
        logger.info("Upload completed ✅")
    else:
        logger.warning("--push_to_hub was not supplied, skipping actual upload. Dry-run only.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("upload-file-to-hf")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local model file to upload.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Destination repository on the Hub (e.g. <user>/<repo>).",
    )
    parser.add_argument(
        "--path_in_repo",
        type=str,
        default=None,
        help="Path inside the repository where the file will be stored (defaults to filename).",
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
