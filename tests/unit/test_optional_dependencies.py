"""Tests for actionable optional-dependency errors."""

import builtins
from unittest.mock import patch

import pytest

from lczerolens.concepts import BinaryConcept, ContinuousConcept, MulticlassConcept
from lczerolens.data import BoardData, GameData, PuzzleData
from lczerolens.model import LczeroModel


def _missing_module(module_name):
    original_import = builtins.__import__

    def import_without_optional_dependency(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"No module named '{module_name}'")
        return original_import(name, *args, **kwargs)

    return import_without_optional_dependency


@pytest.mark.parametrize(
    ("call", "module_name"),
    [
        (lambda: BinaryConcept.compute_metrics([], []), "sklearn"),
        (BinaryConcept.get_dataset_feature, "datasets"),
        (lambda: MulticlassConcept.compute_metrics([], []), "sklearn"),
        (MulticlassConcept.get_dataset_feature, "datasets"),
        (lambda: ContinuousConcept.compute_metrics([], []), "sklearn"),
        (ContinuousConcept.get_dataset_feature, "datasets"),
        (GameData.get_dataset_features, "datasets"),
        (BoardData.get_dataset_features, "datasets"),
        (PuzzleData.get_dataset_features, "datasets"),
        (lambda: LczeroModel.from_hf("lczerolens/maia-1100"), "huggingface_hub"),
    ],
)
def test_missing_optional_dependency_names_the_install_extra(call, module_name):
    """Optional functionality points users to the package extra that enables it."""
    with patch("builtins.__import__", side_effect=_missing_module(module_name)):
        with pytest.raises(ImportError, match=r"pip install lczerolens\[(datasets|hub)\]"):
            call()
