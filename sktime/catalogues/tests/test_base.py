"""Unit tests for the BaseCatalogue class."""

__author__ = ["jgyasu"]

import pytest

from sktime.catalogues.base import BaseCatalogue


class DummyCatalogue(BaseCatalogue):
    """Minimal subclass for testing BaseCatalogue behavior."""

    def _get(self):
        """Return dummy catalogue contents."""
        return {
            "dataset": ["Airline"],
            "metric": ["MeanAbsoluteError"],
        }


class InvalidCatalogue(BaseCatalogue):
    """Catalogue with invalid structure."""

    def _get(self):
        """Return invalid catalogue."""
        return {"dataset": "Airline"}


@pytest.fixture
def dummy_catalogue():
    """Return a minimal dummy catalogue."""
    return DummyCatalogue()


def test_available_categories(dummy_catalogue):
    """available_categories should return catalogue keys."""
    assert set(dummy_catalogue.available_categories()) == {
        "dataset",
        "metric",
    }


def test_get_all_and_specific(dummy_catalogue):
    """get should flatten categories and support filtering."""
    all_items = dummy_catalogue.get("all")

    assert isinstance(all_items, list)
    assert set(all_items) == {"Airline", "MeanAbsoluteError"}

    assert dummy_catalogue.get("dataset") == ["Airline"]


def test_get_invalid_type(dummy_catalogue):
    """Invalid category names should raise KeyError."""
    with pytest.raises(KeyError):
        dummy_catalogue.get("invalid")


def test_get_as_string(dummy_catalogue):
    """get(as_object=False) should return names/specifications."""
    items = dummy_catalogue.get("all", as_object=False)

    assert all(isinstance(item, str) for item in items)


def test_get_as_object_uses_cache(dummy_catalogue):
    """Repeated object resolution should reuse cached objects."""
    first = dummy_catalogue.get("all", as_object=True)
    second = dummy_catalogue.get("all", as_object=True)

    assert first is second


def test_len_and_contains(dummy_catalogue):
    """__len__ and __contains__ should use public catalogue names."""
    assert len(dummy_catalogue) == 2

    assert "Airline" in dummy_catalogue
    assert "MeanAbsoluteError" in dummy_catalogue

    assert "Longley" not in dummy_catalogue


def test_invalid_catalogue_structure_raises():
    """Invalid catalogue structures should raise TypeError."""
    catalogue = InvalidCatalogue()

    with pytest.raises(TypeError, match="must contain a list"):
        catalogue.get("all")
