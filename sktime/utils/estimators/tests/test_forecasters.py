"""Tests for Mock Forecasters."""

__author__ = ["ltsaprounis"]

from copy import deepcopy

import pytest

from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils._testing.deep_equals import deep_equals
from sktime.utils.estimators import MockUnivariateForecasterLogger

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
    """Tests the log of the MockUnivariateForecasterLogger.

    Tests the following:
    - log format and content
    - All the private methods that have logging enabled are in the log
    - the correct inner mtypes are preserved, according to the forecaster tags
    """
    forecaster = MockUnivariateForecasterLogger()
    forecaster.fit(y, X_train, fh)
    forecaster.predict(fh, X_pred)
    forecaster.update(y, X_train, fh)
    forecaster.predict_quantiles(fh=fh, X=X_pred, alpha=[0.1, 0.9])

    _X_train = deepcopy(X_frame_train) if X_train is not None else None
    _X_pred = deepcopy(X_frame_pred) if X_pred is not None else None

    expected_log = [
        ("_fit", {"y": y_series, "X": _X_train, "fh": fh}),
        ("_predict", {"fh": fh, "X": _X_pred}),
        ("_update", {"y": y_series, "X": _X_train, "update_params": fh}),
        ("_predict_quantiles", {"fh": fh, "X": _X_pred, "alpha": [0.1, 0.9]}),
    ]

    equals, msg = deep_equals(forecaster.log, expected_log, return_msg=True)
    assert equals, msg
