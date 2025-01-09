"""
Integration tests using the notebooks.
"""

import subprocess
import pytest

NOTEBOOKS = [
    "docs/source/notebooks/features/convert-official-weights.ipynb",
    "docs/source/notebooks/features/evaluate-models-on-puzzle.ipynb",
    "docs/source/notebooks/features/move-prediction.ipynb",
    "docs/source/notebooks/tutorials/evidence-of-learned-look-ahead.ipynb",
    "docs/source/notebooks/tutorials/piece-value-estimation-using-lrp.ipynb",
    "docs/source/notebooks/tutorials/train-saes.ipynb",
    "docs/source/notebooks/walkthrough.ipynb",
]


def run_notebook(notebook):
    result = subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", notebook],
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)


class TestNotebooks:
    def test_error_notebook(self):
        with pytest.raises(subprocess.CalledProcessError):
            run_notebook("tests/assets/error.ipynb")

    @pytest.mark.parametrize("notebook", NOTEBOOKS)
    def test_notebook(self, notebook):
        run_notebook(notebook)
