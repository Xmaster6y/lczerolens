"""Tests for the release artifacts."""

from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def _build_distributions(output_dir: Path) -> tuple[Path, Path]:
    """Build the wheel and source distribution into ``output_dir``."""
    subprocess.run(
        ["uv", "build", "--out-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return next(output_dir.glob("*.whl")), next(output_dir.glob("*.tar.gz"))


@pytest.mark.slow
def test_distributions_contain_only_library_release_files(tmp_path: Path) -> None:
    """Keep development-only surfaces out of published artifacts."""
    wheel_path, sdist_path = _build_distributions(tmp_path)

    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_files = set(wheel.namelist())

    assert any(path.startswith("lczerolens/") for path in wheel_files)
    assert any(path.endswith(".dist-info/METADATA") for path in wheel_files)
    assert all(path.startswith("lczerolens/") or ".dist-info/" in path for path in wheel_files)

    with tarfile.open(sdist_path) as sdist:
        sdist_files = {
            Path(member.name).relative_to(sdist.getnames()[0].split("/", 1)[0]).as_posix()
            for member in sdist.getmembers()
            if member.isfile()
        }

    allowed_roots = {"LICENSE", "PKG-INFO", "README.md", "pyproject.toml", "setup.cfg"}
    assert all(
        path in allowed_roots or path.startswith("src/lczerolens/") or path.startswith("src/lczerolens.egg-info/")
        for path in sdist_files
    )
