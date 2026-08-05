"""Execute every maintained notebook as an explicit integration contract."""

from pathlib import Path

from nbclient import NotebookClient
import nbformat
import pytest


ROOT = Path(__file__).parents[2]
NOTEBOOK_DIRECTORY = ROOT / "docs" / "source" / "notebooks"
EXPECTED_NOTEBOOKS = {
    "features/chess-evidence.ipynb",
    "features/evaluate-positions.ipynb",
    "features/models-and-inputs.ipynb",
    "features/replayable-search.ipynb",
    "tutorials/analyze-puzzles.ipynb",
    "tutorials/compare-models.ipynb",
    "tutorials/decision-analysis.ipynb",
}
NOTEBOOKS = tuple(sorted(NOTEBOOK_DIRECTORY.rglob("*.ipynb")))


@pytest.mark.integration
def test_maintained_notebook_manifest_is_complete():
    assert {path.relative_to(NOTEBOOK_DIRECTORY).as_posix() for path in NOTEBOOKS} == EXPECTED_NOTEBOOKS


@pytest.mark.integration
@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda path: path.stem)
def test_maintained_notebook_executes(notebook_path):
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
