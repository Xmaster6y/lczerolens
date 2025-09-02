"""Scrape model artefacts from the LCZero storage, optionally upload them to the 🤗 Hub.

Example
-------
```bash
uv run python -m scripts.models.run_scraping \
    --pattern "^512x19-t81-swa-(?P<model_id>\d+)\.pb\.gz\$" \
    --repo_id lczerolens/run-512x19-t81-swa \
    --limit 2
```
"""

import argparse
import re
import shutil
import tempfile
from pathlib import Path

import requests
from huggingface_hub import create_repo, upload_file
from loguru import logger

from scripts.constants import HF_TOKEN

BASE_URL = "https://storage.lczero.org/files/"


def list_remote_files(url: str, pattern: str | None = None) -> list[str]:
    """Return a list of filenames available at the remote LCZero storage.

    Parameters
    ----------
    pattern: str | None
        Optional regular expression to filter filenames. If *None*, return all files.
    """
    logger.info(f"Fetching directory listing from {url}…")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # Basic HTML index, filenames appear between href="<name>"
    filenames = re.findall(r'href="([^\"]+)"', resp.text)

    # Remove parent directory and sub-directory references
    filenames = [f for f in filenames if not f.endswith("/") and f != "../"]

    if pattern is not None:
        regex = re.compile(pattern)
        selected = []
        for fname in filenames:
            m = regex.search(fname)
            if m and "model_id" in m.groupdict():
                selected.append(fname)
        filenames = selected

    logger.info(f"Found {len(filenames)} files{f' matching {pattern}' if pattern else ''}.")
    return filenames


def download_file(filename: str, url: str, dest_dir: Path) -> Path:
    """Download *filename* from LCZero storage into *dest_dir* and return the local path."""
    url = f"{url}{filename}"
    local_path = dest_dir / filename
    logger.info(f"Downloading {url} → {local_path}…")

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)

    logger.info("Download completed ✅")
    return local_path


def push_to_hf(local_path: Path, path_in_repo: str, repo_id: str, push: bool) -> None:
    """Upload *local_path* to *repo_id* at *path_in_repo*.

    The file is stored under `path_in_repo` inside the same repo for all checkpoints.
    """

    if push:
        logger.info(f"Uploading {local_path} to {repo_id}:{path_in_repo}…")
        create_repo(repo_id, token=HF_TOKEN, exist_ok=True, repo_type="model")
        upload_file(path_or_fileobj=str(local_path), path_in_repo=path_in_repo, repo_id=repo_id, token=HF_TOKEN)
        logger.info("Upload completed ✅")
    else:
        logger.warning("--push_to_hub was not supplied, skipping actual upload. Dry-run only.")


def main(args: argparse.Namespace) -> None:
    pattern = args.pattern

    url = BASE_URL + args.url_suffix
    filenames = list_remote_files(url, pattern)
    if args.limit is not None:
        filenames = filenames[: args.limit]
        logger.info(f"Limiting to first {args.limit} files (post-filtering).")

    if not filenames:
        logger.warning("No files to process. Exiting.")
        return
    else:
        all_filenames = "\n".join(filenames)
        logger.info(f"Found the following {len(filenames)} files:\n{all_filenames}")

    if args.download or args.push_to_hub:
        regex = re.compile(pattern)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            logger.info(f"Using temporary directory {tmp_path}")

            for filename in filenames:
                try:
                    match = regex.search(filename)
                    assert match and "model_id" in match.groupdict(), "Regex did not capture 'model_id' group."
                    raw_id = match.group("model_id")
                    model_id = raw_id.zfill(8)

                    # Determine repo ID
                    repo_id = args.repo_id or match.groupdict().get("run", "checkpoints")

                    path_in_repo = f"{model_id}.pb.gz"

                    local_file = download_file(filename, url, tmp_path)
                    push_to_hf(local_file, path_in_repo, repo_id, args.push_to_hub)
                except Exception as e:
                    logger.exception(f"Failed processing {filename}: {e}")
                finally:
                    # Ensure local file removed (even within temp dir)
                    try:
                        local_file.unlink(missing_ok=True)
                    except Exception:
                        pass

            logger.info("All done 🎉")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("scrap-runs")
    parser.add_argument(
        "--url_suffix",
        default="",
        help="Folder to download the files from.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=r"^512x19-t81-swa-(?P<model_id>\d+)\.pb\.gz$",
        help="Regular expression to filter filenames, should contain named group model_id.",
    )
    parser.add_argument(
        "--repo_id",
        default="lczerolens/run-512x19-t81-swa",
        type=str,
        help="Repository on the Hub.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N files (useful for testing).",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Download the files.",
    )
    parser.add_argument(
        "--push_to_hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Actually push to the Hub.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
