import numpy as np
import pandas as pd
import pytest

from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.mapa import MAPAForecaster


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


def test_decompose():
    """Test the `_decompose` method of the MAPAForecaster class.

    Verifies the following:
    - The method correctly handles decomposition for different aggregation levels.
    - The returned decomposed time series matches the input data.
    - The seasonal period and seasonal flag are calculated correctly.
    """

    test_params = MAPAForecaster.get_test_params()

    for params in test_params:
        params["aggregation_levels"] = params.get("aggregation_levels", [1, 2, 4])
        forecaster = MAPAForecaster(**params)
        y = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})

        for level in params["aggregation_levels"]:
            y_decomp, seasonal_enabled, seasonal_period = forecaster._decompose(
                y, level
            )

            # Check if returned data matches input
            assert isinstance(y_decomp, pd.DataFrame)
            assert y_decomp.equals(y)

            # Check seasonal period calculation
            expected_seasonal_period = (
                forecaster.sp // level if (forecaster.sp % level == 0) else 1
            )
            assert seasonal_period == expected_seasonal_period

            # Check if seasonal_enabled is correct
            expected_seasonal = (
                (forecaster.sp % level == 0)
                and (seasonal_period > 1)
                and (len(y) >= 2 * seasonal_period)
                and (level < forecaster.sp)
            )
            assert seasonal_enabled == expected_seasonal


def test_combine_forecasts():
    """Test the `_combine_forecasts` method of the MAPAForecaster class.

    Verifies the following:
    - Forecasts are combined correctly using mean, median, or weighted mean.
    - Invalid combination methods raise an appropriate error.
    """
    forecaster = MAPAForecaster(aggregation_levels=[1, 2, 4])

    forecasts = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

    # Test mean combination
    forecaster.forecast_combine = "mean"
    combined = forecaster._combine_forecasts(forecasts)
    expected_mean = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(combined, expected_mean)

    # Test median combination
    forecaster.forecast_combine = "median"
    combined = forecaster._combine_forecasts(forecasts)
    expected_median = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(combined, expected_median)

    # Test weighted mean combination
    forecaster.forecast_combine = "weighted_mean"
    forecaster.weights = [0.5, 0.3, 0.2]
    combined = forecaster._combine_forecasts(forecasts)
    expected_weighted = np.average(forecasts, axis=0, weights=[0.5, 0.3, 0.2])
    np.testing.assert_array_almost_equal(combined, expected_weighted)

    # Test invalid combination method
    with pytest.raises(ValueError):
        forecaster.forecast_combine = "invalid"
        forecaster._combine_forecasts(forecasts)


def test_aggregate():
    """Test the `_aggregate` method of the MAPAForecaster class.

    Verifies the following:
    - The aggregation is correctly applied for different levels.
    - Different aggregation methods (e.g., sum) produce expected results.
    """
    forecaster = MAPAForecaster(aggregation_levels=[1, 2, 3])
    test_data = pd.DataFrame({"value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})

    # Test level 1 (no aggregation)
    level_1 = forecaster._aggregate(test_data, 1)
    assert level_1.equals(test_data)

    # Test level 2 aggregation
    level_2 = forecaster._aggregate(test_data, 2)
    expected_level_2 = pd.DataFrame({"value": [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]})
    pd.testing.assert_frame_equal(level_2, expected_level_2)

    # Test level 3 aggregation
    level_3 = forecaster._aggregate(test_data, 3)
    expected_level_3 = pd.DataFrame({"value": [2.0, 5.0, 8.0, 11.0]})
    pd.testing.assert_frame_equal(level_3, expected_level_3)

    # Test with different aggregation method
    forecaster = MAPAForecaster(aggregation_levels=[1, 2, 3], agg_method="sum")
    level_2_sum = forecaster._aggregate(test_data, 2)
    expected_level_2_sum = pd.DataFrame({"value": [3, 7, 11, 15, 19, 23]})
    pd.testing.assert_frame_equal(level_2_sum, expected_level_2_sum)


def test_predict(sample_data):
    """Test the `_predict` method of the MAPAForecaster class.

    Verifies the following:
    - The method produces valid predictions for various forecast horizons.
    - Predictions are finite and have the correct shape.
    - Predictions work with different decomposition types and combination methods.
    """
    test_params = MAPAForecaster.get_test_params()

    for params in test_params:
        params["aggregation_levels"] = params.get("aggregation_levels", [1, 2, 4])
        forecaster = MAPAForecaster(**params)

        fh = np.arange(1, 4)
        forecaster.fit(sample_data, fh=fh)

        predictions = forecaster._predict(fh)

        # Basic validation of predictions
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(fh)
        assert predictions.shape[1] == sample_data.shape[1]

        # Check if predictions are finite
        assert np.all(np.isfinite(predictions.values))

        # Test with multiplicative decomposition
        forecaster = MAPAForecaster(**{**params, "decompose_type": "multiplicative"})
        forecaster.fit(sample_data, fh=fh)
        mult_predictions = forecaster._predict(fh)

        assert isinstance(mult_predictions, pd.DataFrame)
        assert len(mult_predictions) == len(fh)
        assert np.all(np.isfinite(mult_predictions.values))

        # Test with different forecast combination methods
        for combine_method in ["mean", "median", "weighted_mean"]:
            forecaster = MAPAForecaster(
                **{**params, "forecast_combine": combine_method}
            )
            forecaster.fit(sample_data, fh=fh)
            predictions = forecaster._predict(fh)

            assert isinstance(predictions, pd.DataFrame)
            assert len(predictions) == len(fh)


def test_predict_with_seasonality(sample_data):
    """Test the `_predict` method with seasonal data.

    Verifies the following:
    - Predictions capture seasonal patterns in the input data.
    - Predictions have the correct shape and are finite.
    """
    forecaster = MAPAForecaster(
        aggregation_levels=[1, 2, 3],
        base_forecaster=ExponentialSmoothing(trend="add", seasonal="add", sp=12),
        sp=12,
    )

    fh = np.arange(1, 13)
    forecaster.fit(sample_data, fh=fh)
    predictions = forecaster._predict(fh)

    # Verify seasonal pattern
    assert isinstance(predictions, pd.DataFrame)
    assert len(predictions) == len(fh)

    # Check if seasonal patterns are present in predictions
    # (basic check - looking for non-linear patterns)
    diffs = np.diff(predictions.values.ravel())
    assert not np.allclose(diffs, np.mean(diffs))  # Should vary due to seasonality


@pytest.mark.parametrize("invalid_level", [-1, 0, 1.5, "2"])
def test_invalid_aggregation_levels(invalid_level):
    """Test that invalid aggregation levels raise appropriate errors."""
    with pytest.raises(ValueError):
        MAPAForecaster(aggregation_levels=[invalid_level])
