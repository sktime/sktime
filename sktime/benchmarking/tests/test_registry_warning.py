"""Tests for _SktimeRegistry duplicate ID warning message."""

__author__ = ["NAME-ASHWANIYADAV"]

import warnings

import pytest

from sktime.benchmarking.benchmarks import _SktimeRegistry
from sktime.classification.dummy import DummyClassifier
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
class TestSktimeRegistryWarning:
    """Test _SktimeRegistry issues correct warning on duplicate entity IDs."""

    def test_duplicate_entity_warning_contains_new_id(self):
        """Test that duplicate registration warning contains the actual new ID.

        Regression test for a bug where the warning message used a plain string
        instead of an f-string, causing the literal text '{entity_id_unique}'
        to appear in the warning instead of the actual unique ID value.
        """
        registry = _SktimeRegistry()
        estimator = DummyClassifier()

        # Register the first entity — no warning expected
        registry.register(entity_id="DummyClassifier", entity=estimator)

        # Register a duplicate — should warn with the actual new unique ID
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry.register(entity_id="DummyClassifier", entity=estimator)

        assert len(caught) == 1, "Expected exactly one warning for duplicate entity"
        warning_message = str(caught[0].message)

        # The warning should contain the actual unique ID, not the literal
        # '{entity_id_unique}' text
        assert "DummyClassifier_2" in warning_message, (
            f"Warning should contain the actual unique ID 'DummyClassifier_2', "
            f"got: {warning_message}"
        )
        assert "{entity_id_unique}" not in warning_message, (
            f"Warning contains literal '{{entity_id_unique}}' instead of the "
            f"actual value — missing f-string prefix. Got: {warning_message}"
        )

    def test_duplicate_entity_warning_is_user_warning(self):
        """Test that the duplicate registration warning is a UserWarning."""
        registry = _SktimeRegistry()
        estimator = DummyClassifier()

        registry.register(entity_id="test_est", entity=estimator)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry.register(entity_id="test_est", entity=estimator)

        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)

    def test_duplicate_entity_gets_unique_id(self):
        """Test that a duplicate entity is stored with a unique ID."""
        registry = _SktimeRegistry()
        estimator = DummyClassifier()

        registry.register(entity_id="MyEstimator", entity=estimator)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            registry.register(entity_id="MyEstimator", entity=estimator)

        assert "MyEstimator" in registry.entities
        assert "MyEstimator_2" in registry.entities
        assert len(registry.entities) == 2
