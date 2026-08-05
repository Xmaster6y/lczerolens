"""Tests for the release artifacts."""

import subprocess
import sys
import tarfile
import zipfile
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
TUTORIAL = ROOT / "examples" / "decision_analysis_tutorial.py"


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
    assert "lczerolens/backends.py" not in wheel_files

    with zipfile.ZipFile(wheel_path) as wheel:
        model_source = wheel.read("lczerolens/model.py").decode()
        metadata_path = next(path for path in wheel_files if path.endswith(".dist-info/METADATA"))
        metadata = wheel.read(metadata_path).decode()
    assert all(
        symbol not in model_source for symbol in ("ForceValue", "PolicyFlow", "ValueFlow", "WdlFlow", "MlhFlow")
    )
    assert "v-lczero-bindings" not in metadata

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


@pytest.mark.slow
def test_built_wheel_runs_maintained_workflow_in_clean_environment(tmp_path: Path) -> None:
    """Install only the wheel, exclude checkout imports, and run all six use cases."""
    if os.environ.get("LCZEROLENS_RUN_WHEEL_TEST") != "1":
        pytest.skip("Set LCZEROLENS_RUN_WHEEL_TEST=1 to run the networked installed-wheel release gate.")
    from examples.decision_analysis_tutorial import TUTORIAL_DECISION_DIGEST
    from lczerolens import DecisionAnalysis

    wheel_path, _ = _build_distributions(tmp_path / "dist")
    environment = tmp_path / "environment"
    subprocess.run(
        ["uv", "venv", "--python", sys.executable, str(environment)],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheel_path)],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )

    probe = subprocess.run(
        [str(python), "-I", "-c", "import lczerolens; print(lczerolens.__file__)"],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    installed_module = Path(probe.stdout.strip()).resolve()
    assert not installed_module.is_relative_to(ROOT)

    artifact = tmp_path / "decision.json"
    completed = subprocess.run(
        [str(python), "-I", str(TUTORIAL), str(artifact)],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    restored = DecisionAnalysis.load(artifact)
    assert restored.digest() == TUTORIAL_DECISION_DIGEST
    assert f"digest={TUTORIAL_DECISION_DIGEST}" in completed.stdout
