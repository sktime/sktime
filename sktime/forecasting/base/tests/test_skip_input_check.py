"""Tests for input_conversion/output_conversion config in vectorized forecasting.

Tests verify that:
1. Conversion configs are correctly managed (off during vectorization, on after).
2. Vectorized fit/predict produces correct results with the optimization.
3. Inner-loop validation is skipped when configs are set to "off".
4. Backward compatibility is maintained for non-vectorized (Series) usage.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["manshusainishab"]

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.naive import NaiveForecaster


@pytest.fixture
def panel_data():
    """Create panel data with 3 instances, 10 time points each."""
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
    """Create univariate series data (non-vectorized path)."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame({"y": np.random.RandomState(42).randn(10)}, index=idx)


def test_input_conversion_config_default():
    """Default input_conversion config should be 'on'."""
    f = NaiveForecaster()
    assert f.get_config()["input_conversion"] == "on"
    assert f.get_config()["output_conversion"] == "on"


def test_conversion_config_reset_after_vectorized_fit(panel_data):
    """Inner clones should have input/output_conversion='on' after fit.

    _vectorize temporarily sets conversion off, then resets to on.
    """
    f = NaiveForecaster()
    f.fit(panel_data, fh=[1, 2, 3])

    assert hasattr(f, "forecasters_")
    for col in f.forecasters_.columns:
        for idx in f.forecasters_.index:
            inner = f.forecasters_.loc[idx, col]
            cfg = inner.get_config()
            assert cfg["input_conversion"] == "on", (
                f"inner ({idx}, {col}): input_conversion not reset after fit"
            )
            assert cfg["output_conversion"] == "on", (
                f"inner ({idx}, {col}): output_conversion not reset after fit"
            )


def test_conversion_config_reset_after_vectorized_predict(panel_data):
    """Inner clones should have conversion='on' after predict."""
    f = NaiveForecaster()
    f.fit(panel_data, fh=[1, 2, 3])
    f.predict()

    for col in f.forecasters_.columns:
        for idx in f.forecasters_.index:
            cfg = f.forecasters_.loc[idx, col].get_config()
            assert cfg["input_conversion"] == "on"
            assert cfg["output_conversion"] == "on"


def test_vectorized_fit_predict_correctness(panel_data):
    """Vectorized fit/predict with NaiveForecaster(strategy='last').

    All instances should produce non-NaN predictions.
    """
    f = NaiveForecaster(strategy="last")
    f.fit(panel_data, fh=[1, 2, 3])
    y_pred = f.predict()

    assert y_pred is not None
    assert len(y_pred) > 0

    if isinstance(y_pred.index, pd.MultiIndex):
        pred_instances = y_pred.index.get_level_values(0).unique()
        data_instances = panel_data.index.get_level_values(0).unique()
        assert set(pred_instances) == set(data_instances)

    assert not y_pred.isna().any().any(), "predictions contain NaN"


def test_inner_fit_uses_conversion_off(panel_data):
    """During vectorized fit, inner _fit calls should see input_conversion='off'.

    Instruments _fit to record the config value at call time.
    """
    f = NaiveForecaster()
    original_fit = type(f)._fit
    recorded_configs = []

    def instrumented_fit(self_inner, y, X, fh):
        recorded_configs.append(self_inner.get_config()["input_conversion"])
        return original_fit(self_inner, y, X, fh)

    with patch.object(type(f), "_fit", instrumented_fit):
        f.fit(panel_data, fh=[1, 2, 3])

    n_instances = len(panel_data.index.get_level_values(0).unique())
    assert len(recorded_configs) == n_instances
    assert all(c == "off" for c in recorded_configs), (
        f"expected all inner _fit calls to see input_conversion='off', "
        f"got {recorded_configs}"
    )


def test_series_path_uses_full_validation(series_data):
    """Non-vectorized Series path should keep conversion='on' throughout."""
    f = NaiveForecaster()
    f.fit(series_data, fh=[1, 2, 3])

    assert f.get_config()["input_conversion"] == "on"
    assert not hasattr(f, "forecasters_") or f.forecasters_ is None

    y_pred = f.predict()
    assert y_pred is not None
    assert len(y_pred) == 3


def test_invalid_data_still_raises():
    """Non-time-series input should still raise with default config."""
    f = NaiveForecaster()
    with pytest.raises((TypeError, ValueError)):
        f.fit("not_a_time_series", fh=[1, 2, 3])


def test_vectorized_update_correctness(panel_data):
    """Vectorized update should work and reset configs afterwards."""
    f = NaiveForecaster()
    f.fit(panel_data, fh=[1, 2, 3])

    instances = []
    for i in range(3):
        idx = pd.date_range("2020-01-11", periods=5, freq="D")
        series = pd.DataFrame(
            {"y": np.random.RandomState(i + 10).randn(5)},
            index=idx,
        )
        series.index.name = "time"
        instances.append(series)

    update_data = pd.concat(instances, keys=[0, 1, 2], names=["instance", "time"])
    f.update(update_data)

    for col in f.forecasters_.columns:
        for idx in f.forecasters_.index:
            cfg = f.forecasters_.loc[idx, col].get_config()
            assert cfg["input_conversion"] == "on"

    y_pred = f.predict()
    assert y_pred is not None
    assert len(y_pred) > 0


def test_set_conversion_config_helper(panel_data):
    """_set_conversion_config_on_forecasters sets configs on all inner clones."""
    f = NaiveForecaster()
    f.fit(panel_data, fh=[1, 2, 3])

    f._set_conversion_config_on_forecasters("off", "off")
    for col in f.forecasters_.columns:
        for idx in f.forecasters_.index:
            cfg = f.forecasters_.loc[idx, col].get_config()
            assert cfg["input_conversion"] == "off"
            assert cfg["output_conversion"] == "off"

    f._set_conversion_config_on_forecasters("on", "on")
    for col in f.forecasters_.columns:
        for idx in f.forecasters_.index:
            cfg = f.forecasters_.loc[idx, col].get_config()
            assert cfg["input_conversion"] == "on"
            assert cfg["output_conversion"] == "on"


def test_set_conversion_config_no_forecasters():
    """Helper should not raise when forecasters_ doesn't exist."""
    f = NaiveForecaster()
    f._set_conversion_config_on_forecasters("off", "off")
    f._set_conversion_config_on_forecasters("on", "on")
