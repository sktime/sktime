# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for native STR, including fixed reference results from R stR."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal, assert_series_equal

from sktime.transformations.detrend import STR


@pytest.mark.parametrize("case,periods", [(1, [3]), (2, [3, 6])])
def test_r_reference(case, periods):
    """Match STRmodel from R for one and multiple time-varying seasonalities."""
    reference = pd.read_csv(
        Path(__file__).parent / "data" / f"str_reference_{case}.csv"
    )
    transformer = STR(
        periods=periods,
        lambdas=[2] + [1.5, 0.3, 0.8] * len(periods),
        trend_knots=6,
        seasonal_time_knots=4,
        seasonal_knots=3,
        return_components=True,
    )
    result = transformer.fit_transform(reference["data"])
    assert_allclose(result, reference.drop(columns="data"), atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("components", [False, True])
@pytest.mark.parametrize("index_type", ["range", "period", "datetime"])
def test_reconstruction(components, index_type):
    """Preserve input labels and reconstruct the observed series exactly."""
    index = {
        "range": pd.RangeIndex(48),
        "period": pd.period_range("2000-01", periods=48, freq="M"),
        "datetime": pd.date_range("2000-01-01", periods=48, freq="D"),
    }[index_type]
    t = np.arange(48)
    X = pd.Series(0.1 * t + np.sin(2 * np.pi * t / 6), index=index, name="signal")
    transformer = STR(periods=[2, 6], return_components=components)
    result = transformer.fit_transform(X)
    assert result.index.equals(X.index)
    assert_allclose(
        transformer.trend_ + transformer.seasonal_.sum(axis=1) + transformer.resid_, X
    )
    restored = transformer.inverse_transform(result)
    if components:
        assert list(restored.columns) == [X.name]
        restored = restored.iloc[:, 0]
    assert_series_equal(restored, X, atol=1e-10, rtol=1e-10)
    if components:
        assert list(result.columns) == ["trend", "resid", "seasonal_2", "seasonal_6"]


def test_linear_trend_and_varying_seasonality():
    """Recover a linear trend and seasonal amplitude that grows linearly."""
    t = np.arange(90)
    trend = 4 + 0.03 * t
    seasonal = (1 + 0.02 * t) * np.sin(2 * np.pi * (t + 1) / 3)
    transformer = STR(periods=3, lambdas=[1, 1, 0.0001, 0.0001])
    transformer.fit(pd.Series(trend + seasonal))
    assert_allclose(transformer.trend_, trend, atol=1e-6)
    assert_allclose(transformer.seasonal_["seasonal_3"], seasonal, atol=1e-6)


def test_transform_new_values_does_not_mutate():
    """Recompute for changed values and return requested indices only."""
    X = pd.Series(np.sin(np.arange(36)), name="signal")
    transformer = STR(periods=3, return_components=True).fit(X)
    original = transformer.transform(X)
    subset = X.iloc[12:24] + 2
    result = transformer.transform(subset)
    assert result.index.equals(subset.index)
    assert_allclose(result.sum(axis=1), subset)
    assert_frame_equal(transformer.transform(X), original)
    changed = X + 1
    assert_allclose(transformer.transform(changed).sum(axis=1), changed)


@pytest.mark.parametrize("components", [False, True])
def test_forecasting_pipeline(components):
    """Both output modes work in a forecasting pipeline."""
    from sktime.forecasting.naive import NaiveForecaster

    X = pd.Series(
        np.sin(np.arange(48)), index=pd.period_range("2000", periods=48, freq="M")
    )
    transformer = STR(periods=[2, 6], return_components=components)
    forecaster = transformer * NaiveForecaster()
    forecaster.fit(X, fh=[1, 2, 7])
    prediction = forecaster.predict()
    assert prediction.index.equals(pd.PeriodIndex([X.index[-1] + h for h in [1, 2, 7]]))
    assert prediction.shape == (3,)
    assert np.isfinite(prediction).all()


def test_inverse_future_repeats_last_cycle():
    """Future inverse adds each fitted season's last cycle at the right phase."""
    X = pd.Series(np.sin(np.arange(48)))
    transformer = STR(periods=[2, 6]).fit(X)
    horizon = pd.Index([48, 49, 54])
    result = transformer.inverse_transform(pd.Series(0.0, index=horizon))
    expected = sum(
        transformer.seasonal_[f"seasonal_{p}"]
        .iloc[-p:]
        .to_numpy()[np.array([0, 1, 6]) % p]
        for p in [2, 6]
    )
    assert_allclose(result, expected)


@pytest.mark.parametrize(
    "params,match",
    [
        ({"periods": []}, "periods"),
        ({"periods": [2, 2]}, "periods"),
        ({"periods": 1}, "periods"),
        ({"periods": 2.5}, "periods"),
        ({"periods": True}, "periods"),
        ({"periods": [3, 30]}, "two full cycles"),
        ({"trend_knots": 2}, "trend_knots"),
        ({"seasonal_time_knots": 2}, "seasonal_time_knots"),
        ({"seasonal_knots": [3, 4]}, "seasonal_knots"),
        ({"seasonal_knots": 1}, "seasonal_knots"),
        ({"lambdas": 0}, "lambdas"),
        ({"lambdas": np.inf}, "lambdas"),
        ({"lambdas": [1, 2]}, "lambdas"),
    ],
)
def test_invalid_parameters(params, match):
    """Reject invalid parameters with actionable messages."""
    with pytest.raises(ValueError, match=match):
        STR(**params).fit(pd.Series(np.arange(48, dtype=float)))


def test_nonfinite_input():
    """Reject infinities before passing them to the linear solver."""
    X = pd.Series(np.arange(48, dtype=float))
    X.iloc[5] = np.inf
    with pytest.raises(ValueError, match="finite"):
        STR(periods=3).fit(X)
