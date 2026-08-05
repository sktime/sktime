# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the deprecation shim of the renamed test framework fixtures.

In sktime 1.2.0, the fixture variables of ``BaseFixtureGenerator`` descendants
were renamed, to align with the ``scikit-base`` test framework:

* ``estimator_class`` -> ``object_class``
* ``estimator_instance`` -> ``object_instance``
* ``estimator_type_filter`` -> ``object_type_filter`` (no alias, see below)

The old fixture names remain available as deprecated aliases until 1.3.0.
"""

__author__ = ["yash-sangwan"]

import pytest

from sktime.forecasting.naive import NaiveForecaster
from sktime.tests.test_all_estimators import BaseFixtureGenerator, QuickTester

# todo 1.3.0: remove this module together with the deprecation aliases
#   in sktime.tests.test_all_estimators.BaseFixtureGenerator


class _OldStyle(BaseFixtureGenerator, QuickTester):
    """Test class using the deprecated fixture names."""

    object_type_filter = "forecaster"

    def test_fixture(self, estimator_instance):
        """Consume the deprecated estimator_instance fixture."""
        assert estimator_instance is not None


class _NewStyle(BaseFixtureGenerator, QuickTester):
    """Test class using the current fixture names."""

    object_type_filter = "forecaster"

    def test_fixture(self, object_instance):
        """Consume the current object_instance fixture."""
        assert object_instance is not None


def test_estimator_type_filter_raises():
    """Check that setting the renamed class attribute raises a directive error."""
    # constructed via type, not a class statement: __init_subclass__ raises during
    # class creation, so the class name is never bound and would only ever be an
    # unused local
    with pytest.raises(TypeError, match="renamed to object_type_filter"):
        type(
            "_Bad",
            (BaseFixtureGenerator,),
            {"estimator_type_filter": "forecaster"},
        )


def test_generator_dict_has_aliases():
    """Check that generator_dict exposes both current and deprecated fixture names."""
    generator_dict = _OldStyle().generator_dict()

    expected = [
        "object_class",
        "object_instance",
        "estimator_class",
        "estimator_instance",
    ]
    for key in expected:
        assert key in generator_dict, f"generator_dict is missing the key {key}"


@pytest.mark.parametrize(
    "old, new",
    [
        ("estimator_class", "object_class"),
        ("estimator_instance", "object_instance"),
    ],
)
def test_alias_generator_deprecation(old, new):
    """Check the alias generators warn, and return the same fixtures as the new ones."""
    generator = _OldStyle()

    with pytest.warns(DeprecationWarning, match=f"{old} .* has been renamed to {new}"):
        _, old_names = getattr(generator, f"_generate_{old}")("test_fixture")

    _, new_names = getattr(generator, f"_generate_{new}")("test_fixture")

    assert old_names == new_names, (
        f"fixtures generated for the deprecated {old} differ from those for {new}"
    )


def test_run_tests_with_deprecated_fixture():
    """Check that run_tests subsets to the passed object for deprecated fixtures.

    Without the alias entries in the temporary generator dict of ``run_tests``,
    a test consuming a deprecated fixture would range over all objects in the
    package, rather than over ``NaiveForecaster`` only.
    """
    results_old = _OldStyle().run_tests(NaiveForecaster, tests_to_run="test_fixture")
    results_new = _NewStyle().run_tests(NaiveForecaster, tests_to_run="test_fixture")

    assert set(results_old) == set(results_new), (
        "run_tests produced different fixtures for deprecated and current names"
    )
    assert set(results_old.values()) == {"PASSED"}, results_old
    assert all("NaiveForecaster" in key for key in results_old), results_old
