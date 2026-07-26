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
        """available_categories should return a list of category names."""
        cats = estimator_instance.available_categories()

        assert isinstance(cats, list)
        assert all(isinstance(cat, str) for cat in cats)

    def test_get_all_returns_names(self, estimator_instance):
        """get('all') should return display names or name dictionaries."""
        items = estimator_instance.get("all")

        assert isinstance(items, list)
        assert all(isinstance(item, (str, dict)) for item in items)

        for item in items:
            if isinstance(item, dict):
                assert all(isinstance(k, str) for k in item)
                assert all(isinstance(v, str) for v in item.values())

    def test_get_by_category(self, estimator_instance):
        """get(category) should return display names for that category."""
        for cat in estimator_instance.available_categories():
            items = estimator_instance.get(cat)

            assert isinstance(items, list)

            for item in items:
                assert isinstance(item, (str, dict))

                if isinstance(item, dict):
                    assert all(isinstance(k, str) for k in item)
                    assert all(isinstance(v, str) for v in item.values())

    def test_get_invalid_category_raises(self, estimator_instance):
        """Unknown categories should raise KeyError."""
        with pytest.raises(KeyError):
            estimator_instance.get("not-a-real-category")

    def test_as_object_returns_objects(self, estimator_instance):
        """as_object=True should return resolved objects."""
        for cat in estimator_instance.available_categories():
            objs = estimator_instance.get(cat, as_object=True)

            assert isinstance(objs, list)

            for obj in objs:
                assert not isinstance(obj, str)

                if isinstance(obj, dict):
                    assert all(not isinstance(v, str) for v in obj.values())

    def test_as_object_caching(self, estimator_instance):
        """Repeated object resolution should use the cache."""
        for cat in estimator_instance.available_categories():
            first = estimator_instance.get(cat, as_object=True)
            second = estimator_instance.get(cat, as_object=True)

            assert first is second

    def test_len_matches_number_of_items(self, estimator_instance):
        """len(catalogue) should equal number of entries returned by get('all')."""
        assert len(estimator_instance) == len(estimator_instance.get("all"))

    def test_contains(self, estimator_instance):
        """Test __contains__ against public catalogue names."""
        items = estimator_instance.get("all")

        for item in items:
            if isinstance(item, dict):
                for key in item:
                    assert key in estimator_instance
            else:
                assert item in estimator_instance

        assert "definitely-not-present" not in estimator_instance
