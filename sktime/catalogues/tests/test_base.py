"""Unit tests for the BaseCatalogue class."""

__author__ = ["jgyasu"]

import pytest

from sktime.catalogues.base import BaseCatalogue


class DummyCatalogue(BaseCatalogue):
    """Minimal subclass for testing BaseCatalogue behavior."""

    def _get(self):
        return {
            "dataset": ["Airline"],
            "metric": ["MeanAbsoluteError"],
        }


@pytest.fixture
def dummy_catalogue():
    """Fixture returning a minimal dummy catalogue."""
    return DummyCatalogue()


def test_available_categories(dummy_catalogue):
    """available_categories should return keys from _get()."""
    assert set(dummy_catalogue.available_categories()) == {"dataset", "metric"}


def test_get_all_and_specific(dummy_catalogue):
    """get() should flatten correctly and handle category filters."""
    all_items = dummy_catalogue.get("all")
    assert isinstance(all_items, list)
    assert set(all_items) == {"Airline", "MeanAbsoluteError"}

    datasets = dummy_catalogue.get("dataset")
    assert datasets == ["Airline"]


def test_get_invalid_type(dummy_catalogue):
    """Invalid object_type should raise KeyError."""
    with pytest.raises(KeyError):
        dummy_catalogue.get("invalid")


def test_get_as_string(dummy_catalogue):
    """get() with as_object=False should return list of string specs."""
    items_as_string = dummy_catalogue.get("all", as_object=False)
    assert all(isinstance(item, str) for item in items_as_string)


def test_get_as_object(dummy_catalogue):
    """get() with as_object=True should return the same items for this dummy."""
    items_as_object = dummy_catalogue.get("all", as_object=True)
    assert all(not isinstance(item, str) for item in items_as_object)


def test_len_and_contains(dummy_catalogue):
    """__len__ and __contains__ should behave correctly."""
    assert len(dummy_catalogue) == 2
    assert "Airline" in dummy_catalogue
    assert "Longley" not in dummy_catalogue
