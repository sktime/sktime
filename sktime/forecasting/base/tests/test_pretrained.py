# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unit tests for BasePretrainedForecaster."""

__author__ = ["Lakshya-2440"]

import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import BasePretrainedForecaster, ForecastingHorizon
from sktime.utils._testing.forecasting import _make_series


class MockPretrainedForecaster(BasePretrainedForecaster):
    """Concrete mock pretrained foundation forecaster for testing."""

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the forecaster."""
        return {"model_name": "mock-foundation-model", "context_length": 32}


@pytest.fixture
def univariate_series():
    """Create a standard univariate daily time series."""
    index = pd.date_range("2023-01-01", periods=20, freq="D")
    return pd.Series(np.arange(20, dtype=np.float64) + 1.0, index=index, name="sales")


@pytest.fixture
def multivariate_df():
    """Create a standard multivariate daily time series DataFrame."""
    index = pd.date_range("2023-01-01", periods=20, freq="D")
    data = {
        "sales": np.arange(20, dtype=np.float64) + 1.0,
        "visitors": np.arange(20, dtype=np.float64) * 2.0 + 10.0,
    }
    return pd.DataFrame(data, index=index)


def test_zero_shot_prediction_without_explicit_fit(univariate_series):
    """Test zero-shot prediction without explicit fit training."""
    forecaster = MockPretrainedForecaster(model_name="test-zero-shot")
    fh = [1, 2, 3]

    # Zero-shot using predict_zero_shot directly without prior fit call
    y_pred = forecaster.predict_zero_shot(y=univariate_series, fh=fh)
    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == 3
    assert forecaster.is_fitted
    assert forecaster.context_series is not None
    assert len(forecaster.context_series) == len(univariate_series)

    # Calling predict directly with y argument
    forecaster2 = MockPretrainedForecaster(model_name="test-zero-shot-2")
    y_pred2 = forecaster2.predict(fh=fh, y=univariate_series)
    assert isinstance(y_pred2, pd.Series)
    assert len(y_pred2) == 3
    assert forecaster2.is_fitted


@pytest.mark.parametrize("y_type", ["series", "dataframe"])
def test_fit_predict_and_predict(y_type, univariate_series, multivariate_df):
    """Test fit, predict, and fit_predict with univariate and multivariate data."""
    y = univariate_series if y_type == "series" else multivariate_df
    forecaster = MockPretrainedForecaster()
    fh = [1, 2, 4]

    # fit followed by predict
    forecaster.fit(y)
    assert forecaster.is_fitted
    y_pred = forecaster.predict(fh=fh)

    if y_type == "series":
        assert isinstance(y_pred, pd.Series)
        assert y_pred.name == y.name
    else:
        assert isinstance(y_pred, pd.DataFrame)
        assert list(y_pred.columns) == list(y.columns)
    assert len(y_pred) == len(fh)

    # fit_predict
    forecaster_fp = MockPretrainedForecaster()
    y_pred_fp = forecaster_fp.fit_predict(y, fh=fh)
    pd.testing.assert_index_equal(y_pred.index, y_pred_fp.index)


@pytest.mark.parametrize(
    "fh_input",
    [
        1,
        [1, 2, 3],
        np.array([2, 5]),
        ForecastingHorizon([1, 3], is_relative=True),
    ],
)
def test_forecasting_horizon_formats_relative(fh_input, univariate_series):
    """Test different relative ForecastingHorizon formats."""
    forecaster = MockPretrainedForecaster()
    forecaster.fit(univariate_series)
    y_pred = forecaster.predict(fh=fh_input)

    expected_len = 1 if isinstance(fh_input, int) else len(fh_input)
    assert len(y_pred) == expected_len
    assert not y_pred.isna().any()


def test_forecasting_horizon_absolute(univariate_series):
    """Test absolute ForecastingHorizon format."""
    forecaster = MockPretrainedForecaster()
    forecaster.fit(univariate_series)

    cutoff = univariate_series.index[-1]
    abs_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=3, freq="D")
    fh_abs = ForecastingHorizon(abs_dates, is_relative=False)

    y_pred = forecaster.predict(fh=fh_abs)
    assert len(y_pred) == 3
    pd.testing.assert_index_equal(y_pred.index, abs_dates)


def test_range_index_support():
    """Test forecaster with integer / RangeIndex time series."""
    y = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.RangeIndex(0, 4), name="val")
    forecaster = MockPretrainedForecaster()
    forecaster.fit(y)

    # Relative fh on integer index
    y_pred = forecaster.predict(fh=[1, 2])
    assert len(y_pred) == 2
    assert list(y_pred.index) == [4, 5]

    # Absolute fh on integer index
    abs_fh = ForecastingHorizon([4, 5, 6], is_relative=False)
    y_pred_abs = forecaster.predict(fh=abs_fh)
    assert len(y_pred_abs) == 3
    assert list(y_pred_abs.index) == [4, 5, 6]


def test_quantile_predictions_univariate(univariate_series):
    """Test quantile predictions on univariate series."""
    forecaster = MockPretrainedForecaster()
    forecaster.fit(univariate_series)
    alphas = [0.1, 0.5, 0.9]

    quantiles = forecaster.predict_quantiles(fh=[1, 2], alpha=alphas)
    assert isinstance(quantiles, pd.DataFrame)
    assert len(quantiles) == 2

    # Check MultiIndex columns (variable, alpha)
    assert isinstance(quantiles.columns, pd.MultiIndex)
    assert list(quantiles.columns.levels[1]) == alphas

    # Monotonicity check across quantiles: q_0.1 <= q_0.5 <= q_0.9
    col = univariate_series.name
    assert (quantiles[(col, 0.1)] <= quantiles[(col, 0.5)]).all()
    assert (quantiles[(col, 0.5)] <= quantiles[(col, 0.9)]).all()


def test_quantile_predictions_multivariate(multivariate_df):
    """Test quantile predictions on multivariate DataFrame."""
    forecaster = MockPretrainedForecaster()
    forecaster.fit(multivariate_df)
    alphas = [0.05, 0.5, 0.95]

    quantiles = forecaster.predict_quantiles(fh=[1, 2], alpha=alphas)
    assert isinstance(quantiles, pd.DataFrame)
    assert len(quantiles) == 2
    for col in multivariate_df.columns:
        assert (col, 0.05) in quantiles.columns
        assert (col, 0.5) in quantiles.columns
        assert (col, 0.95) in quantiles.columns
        assert (quantiles[(col, 0.05)] <= quantiles[(col, 0.95)]).all()


def test_quantile_predictions_zero_shot(univariate_series):
    """Test zero-shot quantile prediction without prior fit."""
    forecaster = MockPretrainedForecaster()
    quantiles = forecaster.predict_quantiles_zero_shot(
        y=univariate_series, fh=[1, 2], alpha=[0.2, 0.8]
    )
    assert isinstance(quantiles, pd.DataFrame)
    assert forecaster.is_fitted
    assert len(quantiles) == 2


def test_context_length_truncation():
    """Test that context_length truncates cached context series to latest points."""
    y = _make_series(n_columns=1, n_timepoints=50)
    forecaster = MockPretrainedForecaster(context_length=15)
    forecaster.fit(y)

    assert len(forecaster.context_series) == 15
    pd.testing.assert_series_equal(forecaster.context_series, y.iloc[-15:])
    assert forecaster.cutoff == y.index[-1]


def test_device_resolution():
    """Test hardware device resolution and fallback."""
    # Explicit CPU device
    fc_cpu = MockPretrainedForecaster(device="cpu")
    assert fc_cpu.device_ == "cpu"

    # Explicit CUDA device
    fc_cuda = MockPretrainedForecaster(device="cuda")
    assert fc_cuda.device_ == "cuda"

    # Fallback when None
    fc_auto = MockPretrainedForecaster(device=None)
    assert fc_auto.device_ in ["cpu", "cuda", "mps"]


def test_custom_predict_zero_shot_override(univariate_series):
    """Test that subclasses can override _predict_zero_shot."""

    class CustomPretrainedForecaster(BasePretrainedForecaster):
        def _predict_zero_shot(self, context, fh=None, X=None):
            fh_abs = fh.to_absolute(self.cutoff)
            idx = fh_abs.to_pandas()
            # Custom linear extrapolation
            last_val = context.iloc[-1]
            return pd.Series(last_val + 100.0, index=idx, name=context.name)

    fc = CustomPretrainedForecaster()
    fc.fit(univariate_series)
    pred = fc.predict(fh=[1, 2])
    assert (pred == univariate_series.iloc[-1] + 100.0).all()
