import numpy as np

from sktime.datasets import load_macroeconomic
from sktime.forecasting.autots import AutoTS
from sktime.forecasting.base import ForecastingHorizon


def test_autots_multivariate_capability():
    # Load the multivariate Macroeconomic dataset (14 series)
    y = load_macroeconomic()

    # Define a relative forecasting horizon (e.g., next 4 periods)
    fh = ForecastingHorizon(np.arange(1, 5), is_relative=True)

    # Instantiate the AutoTS forecaster with minimal configuration for a quick run
    model = AutoTS(max_generations=1, n_jobs=1)

    # Fit the model on the multivariate data
    model.fit(y, fh=fh)

    # Generate forecasts using the fitted model
    forecast = model.predict(fh=fh)

    # Assert that the forecast DataFrame has the correct shape:
    # - Number of rows equals the length of the forecast horizon
    # - Number of columns equals the number of series in the input data
    assert forecast.shape[0] == len(fh), (
        f"Expected forecast length {len(fh)}, got {forecast.shape[0]}"
    )
    assert forecast.shape[1] == y.shape[1], (
        f"Expected forecast to have {y.shape[1]} columns, got {forecast.shape[1]}"
    )

    # Ensure that the forecast values are not entirely NaN
    assert not forecast.isna().all().all(), "Forecast output contains only NaN values"
