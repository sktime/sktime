import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import TrendForecaster


def test_base_raises_error_for_unsupported_weights():
    """Test that forecasters without the sample_weight capability raise an error."""
    y = pd.Series([1, 2, 3], index=pd.period_range("2020-01-01", periods=3, freq="D"))
    weights = np.array([0.1, 0.5, 0.4])

    forecaster = NaiveForecaster()

    msg = "does not support sample_weight"
    with pytest.raises(NotImplementedError, match=msg):
        forecaster.fit(y, sample_weight=weights)


def test_base_passes_weights_if_supported():
    """Test that weights are safely passed to _fit if the capability tag is True."""

    class MockWeightedForecaster(BaseForecaster):
        _tags = {
            "capability:sample_weight": True,
            "requires-fh-in-fit": False,
            "scitype:y": "univariate",
        }

        def _fit(self, y, X=None, fh=None, sample_weight=None):
            self.captured_weight_ = sample_weight
            return self

        def _predict(self, fh, X=None):
            return pd.Series([1] * len(fh), index=fh)

    y = pd.Series([1, 2, 3], index=pd.period_range("2020-01-01", periods=3, freq="D"))
    weights = np.array([0.1, 0.5, 0.4])

    forecaster = MockWeightedForecaster()
    forecaster.fit(y, sample_weight=weights)

    assert hasattr(forecaster, "captured_weight_")
    assert forecaster.captured_weight_ is not None
    np.testing.assert_array_equal(forecaster.captured_weight_, weights)


def test_sample_weights_with_panel_data():
    """Test that sample_weight works with panel (multivariate) time series data."""
    y_panel = pd.DataFrame(
        {"var1": [1.0, 2.0, 3.0, 4.0], "var2": [5.0, 6.0, 7.0, 8.0]},
        index=pd.period_range("2020-01-01", periods=4, freq="D"),
    )

    weights = np.array([0.1, 0.2, 0.3, 0.4])

    forecaster = TrendForecaster()
    forecaster.fit(y_panel, sample_weight=weights)

    assert forecaster._is_fitted


def test_sample_weights_with_univariate_long_series():
    """Test that sample_weight works with longer univariate time series."""
    y_long = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        index=pd.period_range("2020-01-01", periods=10, freq="D"),
    )

    weights = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.2, 0.15, 0.1, 0.05, 0.05])

    forecaster = TrendForecaster()
    forecaster.fit(y_long, sample_weight=weights)

    assert forecaster._is_fitted


def test_sample_weights_with_hierarchical_data():
    """Test that sample_weight works with hierarchical time series data."""
    index = pd.MultiIndex.from_product(
        [["A", "B"], ["x", "y"], pd.period_range("2020-01-01", periods=3, freq="D")],
        names=["level1", "level2", "time"],
    )
    y_hier = pd.DataFrame({"var": np.arange(12.0)}, index=index)

    weights = np.ones(12) * 0.1

    forecaster = TrendForecaster()
    forecaster.fit(y_hier, sample_weight=weights)

    assert forecaster._is_fitted
