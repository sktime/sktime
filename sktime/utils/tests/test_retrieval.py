"""Testing retrieval utilities."""

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.retrieval import _all_classes, _all_functions


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils.retrieval", "sktime.utils.adapters"]),
    reason="Run if relevant content has changed.",
)
def test_all_functions():
    """Test that _all_functions retrieves all functions."""
    res = _all_functions("sktime.utils.adapters")
    names = [name for name, _ in res]

    EXPECTED_NAMES = {
        "_clone_fitted_params",
        "_get_fitted_params_safe",
        "_safe_call",
        "_method_has_arg",
        "_method_has_param_and_default",
    }

    assert EXPECTED_NAMES == set(names)


@pytest.mark.skipif(
    not run_test_module_changed(["sktime.utils.retrieval", "sktime.utils.estimators"]),
    reason="Run if relevant content has changed.",
)
def test_all_classes():
    """Test that _all_classes retrieves all classes."""
    res = _all_classes("sktime.utils.estimators")
    names = [name for name, _ in res]

    assert "MockForecaster" in names
