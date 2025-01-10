"""Lens tests."""

from typing import Any
import pytest

from lczerolens import Lens
from lczerolens.model import LczeroModel


@Lens.register("test_lens")
class TestLens(Lens):
    """Test lens."""

    def is_compatible(self, model: LczeroModel) -> bool:
        return True

    def analyse(self, *inputs, **kwargs) -> Any:
        pass


class TestLensRegistry:
    def test_lens_registry_duplicate(self):
        """Test that registering a lens with an existing name raises an error."""
        with pytest.raises(ValueError, match="Lens .* already registered"):

            @Lens.register("test_lens")
            class DuplicateLens(Lens):
                """Duplicate lens."""

                def is_compatible(self, model: LczeroModel) -> bool:
                    return True

                def analyse(self, *inputs, **kwargs) -> Any:
                    pass

    def test_lens_registry_missing(self):
        """Test that instantiating a non-registered lens raises an error."""
        with pytest.raises(KeyError, match="Lens .* not found"):
            Lens.from_name("non_existent_lens")
