import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.forecasting.mapa import MAPAForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.tests.test_switch import run_test_for_class


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


@pytest.mark.skipif(
    not run_test_for_class(MAPAForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "sp,level,expected_seasonal",
    [
        (6, 1, True),
        (12, 1, True),
        (12, 2, True),
        (12, 4, True),
        (12, 6, True),
        (12, 12, False),
        (12, 3, True),
    ],
)
def test_decompose(sample_data, sp, level, expected_seasonal):
    """Test the `_decompose` method.

    Verifies the following:

    - Decomposition separates trend, seasonal, and residual components.
    - Seasonal component is enabled or disabled as expected.
    - Reconstructed series matches the aggregated input data.
    """
    forecaster = MAPAForecaster(
        aggregation_levels=[level], sp=sp, decompose_type="multiplicative"
    )

    agg_data = forecaster._aggregate(sample_data, level)

    decomposed, seasonal_enabled, seasonal_period = forecaster._decompose(
        agg_data, level
    )

    assert isinstance(decomposed, pd.DataFrame)
    assert seasonal_enabled == expected_seasonal
    assert seasonal_period == sp // level if expected_seasonal else 1

    expected_columns = ["value_trend", "value_seasonal", "value_residual"]
    assert all(col in decomposed.columns for col in expected_columns)

    if forecaster.decompose_type == "multiplicative":
        reconstructed = (
            decomposed["value_trend"]
            * decomposed["value_seasonal"]
            * decomposed["value_residual"]
        )
    else:
        reconstructed = (
            decomposed["value_trend"]
            + decomposed["value_seasonal"]
            + decomposed["value_residual"]
        )

    np.testing.assert_allclose(
        reconstructed.values, agg_data["value"].values, rtol=1e-3, atol=1e-3
    )

    if seasonal_enabled:
        seasonal = decomposed["value_seasonal"].values
        if forecaster.decompose_type == "multiplicative":
            assert np.all(seasonal > 0)

            seasonal_matrix = seasonal[
                : len(seasonal) - (len(seasonal) % seasonal_period)
            ]
            seasonal_matrix = seasonal_matrix.reshape(-1, seasonal_period)
            cv = np.std(seasonal_matrix, axis=0) / np.mean(seasonal_matrix, axis=0)
            assert np.mean(cv) < 1.0
        else:
            seasonal_sum = np.sum(seasonal[:seasonal_period])
            assert abs(seasonal_sum) < np.std(agg_data["value"]) * seasonal_period


@pytest.mark.skipif(
    not run_test_for_class(MAPAForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "combine_method,weights,expected",
    [
        ("mean", None, np.array([2.0, 3.0, 4.0])),
        ("median", None, np.array([2.0, 3.0, 4.0])),
        ("weighted_mean", [0.5, 0.3, 0.2], np.array([1.7, 2.7, 3.7])),
    ],
)
def test_combine_forecasts(combine_method, weights, expected):
    """Test the `_combine_forecasts` method.

    Verifies the following:

    - Correct forecast combination methods are applied.
    - Combined forecasts match expected results for mean, median, and weighted mean.
    """
    forecaster = MAPAForecaster(
        aggregation_levels=[1, 2, 4], forecast_combine=combine_method, weights=weights
    )

    forecasts = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    combined = forecaster._combine_forecasts(forecasts)
    np.testing.assert_array_almost_equal(combined, expected)


@pytest.mark.skipif(
    not run_test_for_class(MAPAForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_combine_forecasts_invalid():
    """Test the `_combine_forecasts` method with invalid combination methods.
    Verifies the following:

    - An error is raised for unsupported combination methods.
    """

    forecaster = MAPAForecaster(
        aggregation_levels=[1, 2, 4], forecast_combine="invalid"
    )
    forecasts = np.array([[1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="Unsupported forecast combination method"):
        forecaster._combine_forecasts(forecasts)


@pytest.mark.skipif(
    not run_test_for_class(MAPAForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "level,agg_method,expected",
    [
        (1, None, pd.DataFrame({"value": range(1, 13)})),
        (1, "mean", pd.DataFrame({"value": range(1, 13)})),
        (2, "mean", pd.DataFrame({"value": [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]})),
        (3, "mean", pd.DataFrame({"value": [2.0, 5.0, 8.0, 11.0]})),
        (2, "sum", pd.DataFrame({"value": [3, 7, 11, 15, 19, 23]})),
    ],
)
def test_aggregate(level, agg_method, expected):
    """Test the `_aggregate` method.

    Verifies the following:

    - Aggregation correctly computes means or sums for various levels.
    - Results match the expected aggregated data.
    """
    forecaster = MAPAForecaster(aggregation_levels=[1, 2, 3], agg_method=agg_method)
    test_data = pd.DataFrame({"value": range(1, 13)})
    result = forecaster._aggregate(test_data, level)
    pd.testing.assert_frame_equal(result, expected)


base_forecaster_params = [
    {},
    {
        "base_forecaster": NaiveForecaster(strategy="mean"),
        "decompose_type": "additive",
    },
    {
        "base_forecaster": PolynomialTrendForecaster(degree=2),
        "decompose_type": "multiplicative",
    },
]

from sktime.utils.dependencies._dependencies import _check_soft_dependencies

if _check_soft_dependencies("statsmodels", severity="none"):
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing

    base_forecaster_params.append(
        {
            "base_forecaster": ExponentialSmoothing(trend="add", seasonal="add", sp=12),
            "decompose_type": "multiplicative",
        }
    )


@pytest.mark.skipif(
    not run_test_for_class(MAPAForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("forecaster_params", base_forecaster_params)
def test_predict(sample_data, forecaster_params):
    """Test the `_predict` method with seasonal data.

    Verifies the following:

    - Predictions have the correct shape and are finite.
    - Predictions are consistent with different forecaster configurations.
    """
    forecaster = MAPAForecaster(aggregation_levels=[1, 2, 4], **forecaster_params)
    fh = ForecastingHorizon(np.arange(1, 4))

    forecaster.fit(sample_data, fh=fh)
    predictions = forecaster._predict(fh)

    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(fh)
    assert predictions.shape[1] == sample_data.shape[1]
    assert np.all(np.isfinite(predictions.values))


@pytest.mark.skipif(
    not run_test_for_class(MAPAForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("invalid_level", [-1, 0, 1.5, "2"])
def test_invalid_aggregation_levels(invalid_level):
    """Test that invalid aggregation levels raise appropriate errors."""
    with pytest.raises(ValueError):
        MAPAForecaster(aggregation_levels=[invalid_level])
