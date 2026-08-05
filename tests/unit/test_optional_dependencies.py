"""Tests for actionable optional-dependency errors."""

import builtins
from unittest.mock import patch

from lczerolens.model import LczeroModel


def _missing_module(module_name):
    original_import = builtins.__import__

    def import_without_optional_dependency(name, *args, **kwargs):
        if name == module_name:
            raise ImportError(f"No module named '{module_name}'")
        return original_import(name, *args, **kwargs)

    return import_without_optional_dependency


def test_missing_hub_dependency_names_the_install_extra():
    """Optional Hub functionality points users to the package extra that enables it."""
    with patch("builtins.__import__", side_effect=_missing_module("huggingface_hub")):
        try:
            LczeroModel.from_hf("lczerolens/maia-1100")
        except ImportError as error:
            assert "pip install lczerolens[hub]" in str(error)
        else:
            raise AssertionError("Missing huggingface_hub must fail with an actionable error.")
