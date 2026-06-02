"""Unit tests common to all catalogues."""

__author__ = ["jgyasu"]
__all__ = []

import pytest

from sktime.tests.test_all_estimators import BaseFixtureGenerator, QuickTester


class CatalogueFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for catalogue tests.

    Fixtures parameterized:
    -----------------------
    estimator_class: catalogue class inheriting from BaseCatalogue
    estimator_instance: instance of catalogue class
    scenario: TestScenario (unused here)
    """

    estimator_type_filter = "catalogue"


class TestAllCatalogues(CatalogueFixtureGenerator, QuickTester):
    """Module-level tests for all sktime catalogues."""

    def test_available_categories_returns_list(self, estimator_instance):
        """Test that available_categories() returns a list of strings."""
        cats = estimator_instance.available_categories()
        assert isinstance(cats, list)
        assert all(isinstance(c, str) for c in cats)

    def test_get_all_returns_list(self, estimator_instance):
        """Test that catalogue.get('all') returns a flat list of names or dicts."""
        items = estimator_instance.get("all")
        assert isinstance(items, list)
        # Items can now be strings or dictionaries (for custom estimator IDs)
        assert all(isinstance(i, (str, dict)) for i in items)

    def test_get_by_category(self, estimator_instance):
        """Test that catalogue.get(category) returns items only from that category."""
        cats = estimator_instance.available_categories()
        for cat in cats:
            items = estimator_instance.get(cat)
            assert isinstance(items, list)
            assert all(isinstance(i, (str, dict)) for i in items)

    def test_get_invalid_category_raises(self, estimator_instance):
        """Test that invalid category names raise KeyError."""
        with pytest.raises(KeyError):
            estimator_instance.get("not-a-real-category")

    def test_as_object_returns_instances(self, estimator_instance):
        """Test `as_object=True` returns instantiated objects or dicts of objects."""
        cats = estimator_instance.available_categories()
        for cat in cats:
            objs = estimator_instance.get(cat, as_object=True)
            assert isinstance(objs, list)
            for o in objs:
                assert not isinstance(o, str)
                if isinstance(o, dict):
                    # Ensure the dictionary values are instantiated objects, not strings
                    assert all(not isinstance(v, str) for v in o.values())

    def test_as_object_caching(self, estimator_instance):
        """Repeated calls to get(as_object=True) should return cached instances."""
        cats = estimator_instance.available_categories()
        for cat in cats:
            first = estimator_instance.get(cat, as_object=True)
            second = estimator_instance.get(cat, as_object=True)
            assert first is second

    def test_len_matches_number_of_items(self, estimator_instance):
        """len(catalogue) should equal number of items in get('all')."""
        items = estimator_instance.get("all")
        assert len(estimator_instance) == len(items)

    def test_contains(self, estimator_instance):
        """Test __contains__ reports presence correctly."""
        items = estimator_instance.get("all")
        for it in items:
            if isinstance(it, dict):
                # The dictionary itself should be found
                assert it in estimator_instance
                # Its individual custom IDs and estimator names should also be found
                for k, v in it.items():
                    assert k in estimator_instance
                    assert v in estimator_instance
            else:
                assert it in estimator_instance
        assert "definitely-not-present" not in estimator_instance
