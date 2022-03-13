# -*- coding: utf-8 -*-
"""Tests for Mock Forecasters."""

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils._testing.deep_equals import deep_equals
from sktime.utils.estimators import MockUnivariateForecaster

y_series = load_airline().iloc[:-5]
y_frame = y_series.to_frame()
X_series_train = load_airline().iloc[:-5]
X_series_pred = load_airline().iloc[-5:]
X_frame_train = X_series_train.to_frame()
X_frame_pred = X_series_pred.to_frame()
fh_absolute = ForecastingHorizon(values=X_series_pred.index, is_relative=False)
fh_relative = ForecastingHorizon(values=[1, 2, 3], is_relative=True)


@pytest.mark.parametrize(
    "y, X_train, X_pred, fh",
    [
        (y_series, X_series_train, X_series_pred, fh_absolute),
        (y_series, X_frame_train, X_frame_pred, fh_absolute),
        (y_series, None, None, fh_absolute),
        (y_series, None, None, fh_relative),
        (y_frame, None, None, fh_relative),
    ],
)
def test_mock_univariate_forecaster_log(y, X_train, X_pred, fh):
    """Tests the log of the MockUnivariateForecaster.

    Tests the following:
    - log format and content
    - All the private methods that have logging enabled are in the log
    - the correct inner mtypes are preserved, according to the forecaster tags
    """
    forecaster = MockUnivariateForecaster()
    forecaster.fit(y, X_train, fh)
    forecaster.predict(fh, X_pred)
    forecaster.update(y, X_train, fh)
    forecaster.predict_quantiles(fh=fh, X=X_pred, alpha=[0.1, 0.9])

    assert deep_equals(
        forecaster.log,
        [
            ("_fit", {"y": y_series, "X": X_frame_train, "fh": fh}),
            ("_predict", {"fh": fh, "X": X_frame_pred}),
            ("_update", {"y": y_series, "X": X_frame_train, "fh": fh}),
            (
                " _predict_quantiles",
                {"fh": fh, "X": X_frame_pred, "alpha": [0.1, 0.9]},
            ),
        ],
    )
