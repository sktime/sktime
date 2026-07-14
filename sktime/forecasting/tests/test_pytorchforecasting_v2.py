# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for pytorch-forecasting v2 adapter interface.

These tests cover:
1. The standalone data conversion function (_sktime_to_ptf_v2_timeseries)
2. Smoke fit/predict tests for PytorchForecastingTFTV2
"""

import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.utils._testing.hierarchical import _make_hierarchical

__author__ = ["vedantag17"]

# Single gate for all tests: skip the whole module if PTF v2 deps are absent.
PTF_V2_DEPS = ["pytorch-forecasting>=1.0.0", "torch", "lightning"]
pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies(*PTF_V2_DEPS, severity="none"),
    reason=(
        "Skipping pytorch-forecasting v2 tests: "
        "pytorch-forecasting>=1.0.0, torch, or lightning not installed."
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Standalone data conversion function (no model needed)
# ──────────────────────────────────────────────────────────────────────────────


def test_sktime_to_ptf_v2_timeseries_panel():
    """Test conversion of sktime panel data to PTF v2 TimeSeries."""
    from sktime.forecasting.base.adapters._pytorchforecasting_v2 import (
        _sktime_to_ptf_v2_timeseries,
    )

    # Create panel data with 3 groups, 20 time points each, 2 columns
    data = _make_hierarchical(
        hierarchy_levels=(3,),
        max_timepoints=20,
        min_timepoints=20,
        n_columns=2,
    )
    y = data["c1"].to_frame()
    X = data[["c0"]]

    timeseries, meta = _sktime_to_ptf_v2_timeseries(y, X)

    # Basic checks
    assert timeseries is not None
    assert len(timeseries) == 3, f"Expected 3 groups, got {len(timeseries)}"
    assert meta["target_name"] == "c1"
    assert meta["index_len"] == 2  # (h0, time)
    assert meta["x_columns"] == ["c0"]

    # Check that __getitem__ returns valid tensors
    sample = timeseries[0]
    assert "y" in sample
    assert "x" in sample
    assert "t" in sample
    assert sample["y"].shape[0] == 20  # 20 timepoints


def test_sktime_to_ptf_v2_timeseries_single():
    """Test conversion of single time series to PTF v2 TimeSeries."""
    from sktime.forecasting.base.adapters._pytorchforecasting_v2 import (
        _sktime_to_ptf_v2_timeseries,
    )

    # Create a single time series (non-hierarchical)
    index = pd.date_range("2000-01-01", periods=30, freq="D")
    y = pd.DataFrame({"target": range(30)}, index=index)

    timeseries, meta = _sktime_to_ptf_v2_timeseries(y)

    assert timeseries is not None
    # Single series = 1 group
    assert len(timeseries) == 1, f"Expected 1 group, got {len(timeseries)}"
    assert meta["target_name"] == "target"
    assert meta["index_len"] == 1


def test_sktime_to_ptf_v2_timeseries_hierarchical():
    """Test conversion of hierarchical data to PTF v2 TimeSeries."""
    from sktime.forecasting.base.adapters._pytorchforecasting_v2 import (
        _sktime_to_ptf_v2_timeseries,
    )

    # Create hierarchical data: 2 levels x 3 groups x 15 timepoints
    data = _make_hierarchical(
        hierarchy_levels=(2, 3),
        max_timepoints=15,
        min_timepoints=15,
        n_columns=2,
    )
    y = data["c1"].to_frame()

    timeseries, meta = _sktime_to_ptf_v2_timeseries(y)

    assert timeseries is not None
    assert len(timeseries) == 6  # 2 * 3 groups
    assert meta["index_len"] == 3  # (h0, h1, time)


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: TFT v2 get_test_params validation
# ──────────────────────────────────────────────────────────────────────────────


def test_tft_v2_get_test_params():
    """Test that get_test_params returns valid parameter dicts."""
    from sktime.forecasting.pytorchforecasting_v2 import PytorchForecastingTFTV2

    params = PytorchForecastingTFTV2.get_test_params()
    assert isinstance(params, list)
    assert len(params) >= 2

    # Each param set should be constructible
    for p in params:
        model = PytorchForecastingTFTV2(**p)
        assert model is not None


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: TFT v2 smoke fit/predict with hierarchical data
# ──────────────────────────────────────────────────────────────────────────────


def test_tft_v2_fit_predict_hierarchical():
    """Smoke test: fit and predict TFT v2 on hierarchical data."""
    from sktime.forecasting.pytorchforecasting_v2 import PytorchForecastingTFTV2

    # Generate hierarchical data: 2 groups, 30 timepoints, 2 columns
    data = _make_hierarchical(
        hierarchy_levels=(2, 5),
        max_timepoints=30,
        min_timepoints=30,
        n_columns=2,
    )
    y = data["c1"].to_frame()

    max_prediction_length = 3
    fh = ForecastingHorizon(range(1, max_prediction_length + 1), is_relative=True)

    model = PytorchForecastingTFTV2(
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 2,
            "enable_checkpointing": False,
            "logger": False,
        },
        data_module_params={
            "context_length": 5,
            "batch_size": 2,
        },
        random_log_path=True,
    )

    model.fit(y=y, fh=fh)
    y_pred = model.predict(fh=fh)

    # Verify output shape and structure
    assert y_pred is not None
    assert isinstance(y_pred, (pd.DataFrame, pd.Series))
    # 2 * 5 = 10 groups * 3 prediction steps = 30 rows
    expected_rows = 10 * max_prediction_length
    assert len(y_pred) == expected_rows, (
        f"Expected {expected_rows} rows, got {len(y_pred)}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: TFT v2 smoke fit/predict with univariate single series
# ──────────────────────────────────────────────────────────────────────────────


def test_tft_v2_fit_predict_univariate():
    """Smoke test: fit and predict TFT v2 on single univariate series."""
    from sktime.forecasting.pytorchforecasting_v2 import PytorchForecastingTFTV2

    # Create a single univariate time series
    index = pd.date_range("2000-01-01", periods=30, freq="D")
    y = pd.DataFrame({"target": range(30)}, index=index)

    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    model = PytorchForecastingTFTV2(
        trainer_params={
            "max_epochs": 1,
            "limit_train_batches": 2,
            "enable_checkpointing": False,
            "logger": False,
        },
        data_module_params={
            "context_length": 5,
            "batch_size": 2,
        },
        random_log_path=True,
    )

    model.fit(y=y, fh=fh)
    y_pred = model.predict(fh=fh)

    assert y_pred is not None
    assert len(y_pred) == 3
