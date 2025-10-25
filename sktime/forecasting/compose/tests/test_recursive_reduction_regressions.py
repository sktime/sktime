# Tests for regression scenarios introduced by speedup of RecursiveReductionForecaster
# - gappy forecasting horizon handling (local/global with/without X)
# - fallback when no effective training rows
#      (estimator_ becomes Series -> constant mean)
# - absence of sklearn feature-name warnings during predict
#
# These target behaviours of the optimized _predict_out_of_sample paths (v2) and
# the v1 fallback logic.

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose._reduce import RecursiveReductionForecaster


@pytest.mark.parametrize("with_X", [False, True])
def test_gappy_fh_local(with_X):
    """Local pooling returns only requested gappy fh steps and preserves order."""
    # create simple univariate series with PeriodIndex
    idx = pd.period_range("2000-01", periods=12, freq="M")
    y = pd.DataFrame({"y": np.arange(12, dtype=float)}, index=idx)

    X = None
    if with_X:
        X = pd.DataFrame({"ex": np.arange(12, dtype=float)}, index=idx)

    fh = ForecastingHorizon([1, 3, 5], is_relative=True)

    f = RecursiveReductionForecaster(
        estimator=LinearRegression(), window_length=4, pooling="local"
    )
    f.fit(y, X=X)
    y_pred = f.predict(fh=fh, X=X)

    # expected absolute index
    expected_index = fh.to_absolute(idx[-1]).to_pandas()
    assert list(y_pred.index) == list(expected_index), (
        "gappy fh not preserved in local path"
    )
    assert y_pred.shape[0] == len(fh)


@pytest.mark.parametrize("with_X", [False, True])
def test_gappy_fh_global(with_X):
    """Global pooling returns gappy fh steps per series and supports X fallback."""
    # build two series in MultiIndex
    series_ids = ["s1", "s2"]
    per_series = 10
    times = pd.period_range("2010-01", periods=per_series, freq="M")
    mi = pd.MultiIndex.from_product([series_ids, times], names=["series", "time"])
    y = pd.DataFrame(
        {"y": np.tile(np.arange(per_series, dtype=float), len(series_ids))}, index=mi
    )

    X = None
    if with_X:
        X = pd.DataFrame(
            {"ex": np.tile(np.arange(per_series, dtype=float), len(series_ids))},
            index=mi,
        )

    fh = ForecastingHorizon([1, 3, 5], is_relative=True)

    f = RecursiveReductionForecaster(
        estimator=LinearRegression(), window_length=3, pooling="global"
    )
    f.fit(y, X=X)
    y_pred = f.predict(fh=fh, X=X)

    # For each series id, ensure only the requested fh steps are present
    for sid in series_ids:
        sid_pred = y_pred.loc[sid]
        expected_idx = fh.to_absolute(times[-1]).to_pandas()
        assert list(sid_pred.index) == list(expected_idx), (
            f"gappy fh not preserved for series {sid}"
        )


def test_constant_mean_fallback():
    """Handles case of no training rows with full lags, estimator_ is Series.

    v2 falls back to v1"""

    # y length too short relative to window_length so that lagged feature matrix empties
    idx = pd.period_range("2021-01", periods=2, freq="M")
    y = pd.DataFrame({"y": [10.0, 10.0]}, index=idx)
    fh = ForecastingHorizon([1, 2], is_relative=True)

    f = RecursiveReductionForecaster(
        estimator=LinearRegression(), window_length=10, pooling="local"
    )
    f.fit(y)
    # ensure estimator_ is a Series of means
    assert isinstance(f.estimator_, pd.Series)

    y_pred = f.predict(fh=fh)
    assert (y_pred.values == 10.0).all(), (
        "Constant mean fallback did not produce mean values"
    )


def test_no_feature_name_warning():
    """Predict should not emit sklearn feature-name warnings (DataFrame wrapping)."""
    idx = pd.period_range("2005-01", periods=15, freq="M")
    y = pd.DataFrame({"y": np.sin(np.arange(15))}, index=idx)
    X = pd.DataFrame({"ex": np.cos(np.arange(15))}, index=idx)
    fh = ForecastingHorizon([1, 2, 4], is_relative=True)

    f = RecursiveReductionForecaster(
        estimator=LinearRegression(), window_length=5, pooling="local"
    )
    f.fit(y, X=X)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = f.predict(fh=fh, X=X)
        msgs = [str(x.message) for x in w]
        offending = [m for m in msgs if "feature names" in m.lower()]
        assert not offending, f"Unexpected feature-name warnings: {offending}"
