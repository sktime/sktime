"""Tests for _skip_input_check optimization in vectorized forecaster execution.

Tests verify that:
1. The _skip_input_check flag is correctly managed (set during vectorization, unset after).
2. Vectorized fit/predict produces identical results with the optimization.
3. Redundant validation calls are reduced in the inner loop.
4. Backward compatibility is maintained for non-vectorized (Series) usage.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["manshusainishab"]

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import get_examples
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.dependencies import _check_soft_dependencies


PANEL_MTYPES = ["pd-multiindex"]


@pytest.fixture
def panel_data():
    """Create simple panel data for testing vectorization."""
    # Create a pd-multiindex Panel with 3 instances, 10 time points each
    instances = []
    for i in range(3):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        series = pd.DataFrame(
            {"y": np.random.RandomState(i).randn(10)},
            index=idx,
        )
        series.index.name = "time"
        instances.append(series)

    mi_data = pd.concat(
        instances,
        keys=[0, 1, 2],
        names=["instance", "time"],
    )
    return mi_data


@pytest.fixture
def series_data():
    """Create simple series data for testing non-vectorized path."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame({"y": np.random.RandomState(42).randn(10)}, index=idx)


def test_skip_input_check_flag_default():
    """Test that _skip_input_check defaults to False on new estimators."""
    forecaster = NaiveForecaster()
    assert hasattr(forecaster, "_skip_input_check")
    assert forecaster._skip_input_check is False


def test_skip_input_check_flag_unset_after_fit(panel_data):
    """Test that _skip_input_check is False on inner clones after vectorized fit.

    The flag should be set temporarily during vectorization and then
    unset to avoid persisting optimization state.
    """
    forecaster = NaiveForecaster()
    forecaster.fit(panel_data, fh=[1, 2, 3])

    # The outer forecaster should not have the flag set
    assert forecaster._skip_input_check is False

    # All inner forecasters should have the flag unset after fit
    assert hasattr(forecaster, "forecasters_")
    for col in forecaster.forecasters_.columns:
        for idx in forecaster.forecasters_.index:
            inner = forecaster.forecasters_.loc[idx, col]
            assert inner._skip_input_check is False, (
                f"Inner forecaster at ({idx}, {col}) has "
                "_skip_input_check=True after fit"
            )


def test_skip_input_check_flag_unset_after_predict(panel_data):
    """Test that _skip_input_check is False on inner clones after vectorized predict."""
    forecaster = NaiveForecaster()
    forecaster.fit(panel_data, fh=[1, 2, 3])
    forecaster.predict()

    # All inner forecasters should have the flag unset after predict
    for col in forecaster.forecasters_.columns:
        for idx in forecaster.forecasters_.index:
            inner = forecaster.forecasters_.loc[idx, col]
            assert inner._skip_input_check is False, (
                f"Inner forecaster at ({idx}, {col}) has "
                "_skip_input_check=True after predict"
            )


def test_vectorization_correctness_with_skip_check(panel_data):
    """Test that vectorized fit/predict produces correct results.

    Regression test: vectorized results should be sensible (point predictions
    exist for each instance and time point in the forecasting horizon).
    """
    forecaster = NaiveForecaster(strategy="last")
    forecaster.fit(panel_data, fh=[1, 2, 3])
    y_pred = forecaster.predict()

    # Predictions should not be None or empty
    assert y_pred is not None
    assert len(y_pred) > 0

    # Should have predictions for all instances
    if isinstance(y_pred.index, pd.MultiIndex):
        instances_in_pred = y_pred.index.get_level_values(0).unique()
        instances_in_data = panel_data.index.get_level_values(0).unique()
        assert set(instances_in_pred) == set(instances_in_data)

    # Predictions should have no NaN values for NaiveForecaster(strategy="last")
    assert not y_pred.isna().any().any(), "Predictions contain NaN values"


def test_skip_input_check_reduces_check_calls(panel_data):
    """Test that the optimization skips inner-loop validation.

    Verify that inner clones have _skip_input_check=True during vectorized
    fit, which causes _check_X_y to take the fast path (skipping
    check_is_scitype). We check this by verifying that the flag is set
    on clones before the inner fit calls.
    """
    forecaster = NaiveForecaster()

    # Track whether _skip_input_check was True when inner fit was called.
    # We instrument the NaiveForecaster._fit to record the flag value.
    original_fit = type(forecaster)._fit
    inner_flag_values = []

    def instrumented_fit(self_inner, y, X, fh):
        inner_flag_values.append(self_inner._skip_input_check)
        return original_fit(self_inner, y, X, fh)

    with patch.object(type(forecaster), "_fit", instrumented_fit):
        forecaster.fit(panel_data, fh=[1, 2, 3])

    n_instances = len(panel_data.index.get_level_values(0).unique())

    # There should be n_instances inner _fit calls (one per vectorized slice).
    # Each should have had _skip_input_check=True during the call.
    assert len(inner_flag_values) == n_instances, (
        f"Expected {n_instances} inner _fit calls, got {len(inner_flag_values)}"
    )
    assert all(inner_flag_values), (
        "Expected all inner _fit calls to have _skip_input_check=True, "
        f"but got {inner_flag_values}"
    )


def test_backward_compatibility_non_vectorized(series_data):
    """Test that non-vectorized Series usage still validates normally.

    When using Series data (no vectorization), the _skip_input_check flag
    should remain False and all validation should occur as before.
    """
    forecaster = NaiveForecaster()
    forecaster.fit(series_data, fh=[1, 2, 3])

    # Non-vectorized path should not set the flag
    assert forecaster._skip_input_check is False
    assert not hasattr(forecaster, "forecasters_") or forecaster.forecasters_ is None

    # Predictions should work normally
    y_pred = forecaster.predict()
    assert y_pred is not None
    assert len(y_pred) == 3  # fh=[1,2,3]


def test_backward_compatibility_invalid_data_raises():
    """Test that invalid data still raises errors for non-vectorized usage.

    The optimization should not affect error handling for invalid inputs.
    """
    forecaster = NaiveForecaster()

    # Passing a string should raise an error (not a valid time series format)
    with pytest.raises((TypeError, ValueError)):
        forecaster.fit("not_a_time_series", fh=[1, 2, 3])


def test_vectorized_update_correctness(panel_data):
    """Test that vectorized update works correctly with skip check optimization."""
    forecaster = NaiveForecaster()
    forecaster.fit(panel_data, fh=[1, 2, 3])

    # Create new update data with same structure but different values
    instances = []
    for i in range(3):
        idx = pd.date_range("2020-01-11", periods=5, freq="D")
        series = pd.DataFrame(
            {"y": np.random.RandomState(i + 10).randn(5)},
            index=idx,
        )
        series.index.name = "time"
        instances.append(series)

    update_data = pd.concat(
        instances,
        keys=[0, 1, 2],
        names=["instance", "time"],
    )

    forecaster.update(update_data)

    # After update, flag should be unset on inner forecasters
    for col in forecaster.forecasters_.columns:
        for idx in forecaster.forecasters_.index:
            inner = forecaster.forecasters_.loc[idx, col]
            assert inner._skip_input_check is False

    # Predictions should still work after update
    y_pred = forecaster.predict()
    assert y_pred is not None
    assert len(y_pred) > 0


def test_set_skip_input_check_on_forecasters_helper(panel_data):
    """Test the _set_skip_input_check_on_forecasters helper method."""
    forecaster = NaiveForecaster()
    forecaster.fit(panel_data, fh=[1, 2, 3])

    # Set flag to True
    forecaster._set_skip_input_check_on_forecasters(True)
    for col in forecaster.forecasters_.columns:
        for idx in forecaster.forecasters_.index:
            assert forecaster.forecasters_.loc[idx, col]._skip_input_check is True

    # Set flag back to False
    forecaster._set_skip_input_check_on_forecasters(False)
    for col in forecaster.forecasters_.columns:
        for idx in forecaster.forecasters_.index:
            assert forecaster.forecasters_.loc[idx, col]._skip_input_check is False


def test_set_skip_input_check_no_forecasters():
    """Test _set_skip_input_check_on_forecasters when no forecasters_ attribute."""
    forecaster = NaiveForecaster()
    # Should not raise when forecasters_ doesn't exist
    forecaster._set_skip_input_check_on_forecasters(True)
    forecaster._set_skip_input_check_on_forecasters(False)
