import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.thief import THieFForecaster


@pytest.fixture
def sample_data():
    """Create sample time series data for testing.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing a sample time series with a sinusoidal pattern
        and some added noise.
    """
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=24, freq="M")
    data = np.sin(np.linspace(0, 4 * np.pi, 24)) * 10 + np.random.normal(0, 1, 24) + 20
    return pd.DataFrame(data, index=dates, columns=["value"])


@pytest.mark.parametrize(
    "index, expected_levels",
    [
        (pd.date_range(start="2020-01-01", periods=12, freq="M"), [1, 2, 3, 4, 6, 12]),
        (
            pd.date_range(start="2020-01-01", periods=52, freq="W"),
            [1, 2, 4, 13, 26, 52],
        ),
        (
            pd.date_range(start="2020-01-01", periods=24, freq="D"),
            [1, 2, 3, 4, 6, 8, 12, 24],
        ),
    ],
)
def test_determine_aggregation_levels(index, expected_levels):
    """Test the `_determine_aggregation_levels` method.

    Verifies the following:

    - Aggregation levels are correctly determined based on the frequency of the
    time series.
    - Levels are factors of seasonal period m.
    """
    y = pd.Series(range(len(index)), index=index)

    forecaster = THieFForecaster(base_forecaster=None)
    levels = forecaster._determine_aggregation_levels(y)

    assert levels == expected_levels


def test_fit(sample_data):
    """Test the fit method of MAPAForecaster."""
    forecaster = THieFForecaster(
        base_forecaster=NaiveForecaster(), reconciliation_method="ols"
    )
    fh = ForecastingHorizon(np.arange(1, 5))
    forecaster.fit(sample_data, fh=fh)
    y_pred = forecaster.predict(fh)

    assert isinstance(y_pred, pd.Series)
    assert len(y_pred) == len(fh)
    assert np.all(np.isfinite(y_pred.values))


@pytest.mark.parametrize(
    "reconciliation_method, expected_output",
    [
        ("bu", np.array([21.21467, 21.21467, 21.21467])),
        ("ols", np.array([16.8999, 16.8999, 16.8999])),
        ("wls_str", np.array([16.8443, 16.8443, 16.8443])),
    ],
)
def test_thief_forecast_matches_r_thief(
    sample_data, reconciliation_method, expected_output
):
    """Test that THieFForecaster predictions match expected outputs for different
    reconciliation methods."""

    fh = ForecastingHorizon(np.arange(1, 4))
    forecaster = THieFForecaster(
        base_forecaster=NaiveForecaster(strategy="last"),
        reconciliation_method=reconciliation_method,
    )

    forecaster.fit(sample_data, fh=fh)
    y_pred = forecaster.predict(fh)

    np.testing.assert_allclose(y_pred.values, expected_output, rtol=1e-3, atol=1e-3)
