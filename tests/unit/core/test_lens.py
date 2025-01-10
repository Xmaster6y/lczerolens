"""Lens tests."""

import pytest

from lczerolens import Lens


@Lens.register("test_lens")
class TestLens(Lens):
    """Test lens."""


class TestLensRegistry:
    def test_lens_registry_duplicate(self):
        """Test that registering a lens with an existing name raises an error."""
        with pytest.raises(ValueError, match="Lens .* already registered"):

            @Lens.register("test_lens")
            class DuplicateLens(Lens):
                """Duplicate lens."""

    def test_lens_registry_missing(self, tiny_model):
        """Test that instantiating a non-registered lens raises an error."""
        with pytest.raises(KeyError, match="Lens .* not found"):
            Lens.from_model("non_existent_lens", tiny_model)
