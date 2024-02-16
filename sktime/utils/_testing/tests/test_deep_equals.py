from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.deep_equals import deep_equals


def test_deep_equals_forecasting_horizon_mismatches_values_length():
    assert not deep_equals(
        ForecastingHorizon(values=[1, 2, 3, 4]), ForecastingHorizon([1, 2, 3])
    )
    assert not deep_equals(
        ForecastingHorizon(values=[1, 2, 3, 4]), ForecastingHorizon([1, 2, 3, 4, 5])
    )
