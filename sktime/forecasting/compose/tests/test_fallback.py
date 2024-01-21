#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests for the FallbackForecaster functionality.

Tests cover the basic operations of the FallbackForecaster, ensuring proper
functionality of fitting, predicting, updating, and handling of errors in the
forecasting process.
"""

__author__ = ["ninedigits"]

import pandas as pd
import pytest

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.compose import EnsembleForecaster, FallbackForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.utils._testing.forecasting import make_forecasting_problem


class ForecastingError(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes
    ----------
    message : str
        Explanation of the error.

    Methods
    -------
    __init__(message)
        Constructs the ForecastingError with the provided message."""

    def __init__(self, message):
        self.message = message
        super().__init__(message)


class DummyForecaster(_HeterogenousMetaEstimator, BaseForecaster):
    """Dummy forecaster used for testing the FallbackForecaster.

    This forecaster is intentionally designed to fail at specified stages of the
    forecasting process (fit, predict, or update) to test the robustness and fallback
    mechanisms of the FallbackForecaster.

    Parameters
    ----------
    raise_at : str
        Stage at which the forecaster should fail. Options are "fit", "predict",
        "update".

    Raises
    ------
    AttributeError
        If `raise_at` is not one of the valid options.
    """

    def __init__(self, raise_at="fit"):
        super().__init__()
        __valid__ = ["fit", "predict", "update"]
        if raise_at not in __valid__:
            raise AttributeError(f"`raise_at` must choose from {__valid__}")
        self.forecaster = NaiveForecaster()
        self.raise_at = raise_at
        self._is_fitted = False

    def _fit(self, y, X=None, fh=None):
        """Fit to training data. Optionally fail here."""
        if self.raise_at == "fit":
            raise ForecastingError("Intentional failure in fit.")
        self.forecaster.fit(y, X, fh)
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None):
        """Make predictions. Optionally fail here."""
        if not self._is_fitted:
            raise ForecastingError("The forecaster is not fitted yet.")
        if self.raise_at == "predict":
            raise ForecastingError("Intentional failure in predict.")
        return self.forecaster.predict(fh, X)

    def _update(self, y, X=None, update_params=True):
        """Update the forecaster. Optionally fail here."""
        if self.raise_at == "update":
            raise ForecastingError("Intentional failure in update.")
        self.forecaster.update(y, X, update_params)
        return self


def test_raises_at_fit():
    """Test dummy forecaster raises at fit"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster = DummyForecaster(raise_at="fit")
    with pytest.raises(ForecastingError):
        forecaster.fit(y=y, fh=[1, 2, 3])


def test_raises_at_predict():
    """Test dummy forecaster raises at predict"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster = DummyForecaster(raise_at="predict")
    forecaster.fit(y=y, fh=[1, 2, 3])
    with pytest.raises(ForecastingError):
        forecaster.predict()


def test_raises_at_update():
    """Test dummy forecaster raises at update"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster = DummyForecaster(raise_at="update")
    forecaster.fit(y=y, fh=[1, 2, 3])
    forecaster.predict()
    with pytest.raises(ForecastingError):
        forecaster.update(y)


def test_fallbackforecaster_fails_at_fit():
    """Test FallbackForecaster fails at first fit, second forecaster succeeds"""
    y = make_forecasting_problem(random_state=42)
    ensemble_forecaster_1 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("fail_fit", DummyForecaster(raise_at="fit")),
        ]
    )
    ensemble_forecaster_2 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("naive", NaiveForecaster()),
        ]
    )
    forecaster = FallbackForecaster(
        [
            (
                "ensemble1_fails_at_fit",
                ensemble_forecaster_1,
            ),
            (
                "ensemble2",
                ensemble_forecaster_2,
            ),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()
    y_pred_expected = ensemble_forecaster_2.predict()
    name = forecaster.current_name
    assert name == "ensemble2"
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)


def test_fallbackforecaster_fails_at_predict():
    """Test FallbackForecaster fails at predict, second forecaster succeeds"""
    y = make_forecasting_problem(random_state=42)
    ensemble_forecaster_1 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("raise_at_predict", DummyForecaster(raise_at="predict")),
        ]
    )
    ensemble_forecaster_2 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("naive", NaiveForecaster()),
        ]
    )
    forecaster = FallbackForecaster(
        [
            (
                "ensemble1_fails_at_predict",
                ensemble_forecaster_1,
            ),
            (
                "ensemble2",
                ensemble_forecaster_2,
            ),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()
    y_pred_expected = ensemble_forecaster_2.predict()
    name = forecaster.current_name
    assert name == "ensemble2"
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)


def test_fallbackforecaster_fails_twice():
    """First two FallbackForecasters fail, third succeeds"""
    y = make_forecasting_problem(random_state=42)
    forecaster_1 = EnsembleForecaster(
        [
            ("trend_0", PolynomialTrendForecaster()),
            ("fails_at_fit_0", DummyForecaster(raise_at="fit")),
        ]
    )
    forecaster_2 = EnsembleForecaster(
        [
            ("trend_1", PolynomialTrendForecaster()),
            ("fails_at_predict_1", DummyForecaster(raise_at="predict")),
        ]
    )
    forecaster_3 = EnsembleForecaster(
        [
            ("trend_2", PolynomialTrendForecaster()),
            ("naive_2", NaiveForecaster()),
        ]
    )
    forecaster_4 = PolynomialTrendForecaster()
    forecaster = FallbackForecaster(
        [
            (
                "ensemble1_fails_at_fit",
                forecaster_1,
            ),
            (
                "ensemble2_fails_at_predict",
                forecaster_2,
            ),
            (
                "ensemble3_succeeds",
                forecaster_3,
            ),
            ("forecaster4_isnt_called", forecaster_4),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()

    forecaster_3.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster_3.predict()
    name = forecaster.current_name
    assert name == "ensemble3_succeeds"
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)


def test_fallbackforecaster_fails_fit_twice():
    """First two FallbackForecasters fail at fit step, third forecaster succeeds"""
    y = make_forecasting_problem(random_state=42)
    ensemble_forecaster_1 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("fails", DummyForecaster(raise_at="fit")),
        ]
    )
    ensemble_forecaster_2 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("fails", DummyForecaster(raise_at="fit")),
        ]
    )
    ensemble_forecaster_3 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("naive", NaiveForecaster()),
        ]
    )
    forecaster = FallbackForecaster(
        [
            (
                "ensemble1",
                ensemble_forecaster_1,
            ),
            (
                "ensemble2",
                ensemble_forecaster_2,
            ),
            (
                "ensemble3",
                ensemble_forecaster_3,
            ),
            ("forecaster4", PolynomialTrendForecaster()),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()
    y_pred_expected = ensemble_forecaster_3.predict()
    name = forecaster.current_name
    assert name == "ensemble3"
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)


def test_all_forecasters_fail1():
    """All forecasters fail; predict and fit"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster1 = ("raise_predict1", DummyForecaster(raise_at="predict"))
    forecaster2 = ("raise_fit1", DummyForecaster(raise_at="fit"))
    forecaster3 = ("raise_fit2", DummyForecaster(raise_at="fit"))
    forecaster = FallbackForecaster([forecaster1, forecaster2, forecaster3])
    with pytest.raises(RuntimeError):
        forecaster.fit(y=y, fh=[1, 2, 3])
        forecaster.predict()


def test_all_forecasters_fail2():
    """All forecasters fail at fit step"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster1 = ("raise_fit1", DummyForecaster(raise_at="fit"))
    forecaster2 = ("raise_fit2", DummyForecaster(raise_at="fit"))
    forecaster3 = ("raise_fit3", DummyForecaster(raise_at="fit"))
    forecaster = FallbackForecaster([forecaster1, forecaster2, forecaster3])
    with pytest.raises(RuntimeError):
        forecaster.fit(y=y, fh=[1, 2, 3])


def test_all_forecasters_fail3():
    """All forecasters fail at predict"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster1 = ("raise_predict1", DummyForecaster(raise_at="predict"))
    forecaster2 = ("raise_predict2", DummyForecaster(raise_at="predict"))
    forecaster3 = ("raise_predict3", DummyForecaster(raise_at="predict"))
    forecaster = FallbackForecaster([forecaster1, forecaster2, forecaster3])
    with pytest.raises(RuntimeError):
        forecaster.fit(y=y, fh=[1, 2, 3])
        forecaster.predict()


def test_many_forecasters_fail1():
    """All forecasters fail at predict"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster1 = ("raise_predict1", DummyForecaster(raise_at="predict"))
    forecaster2 = ("raise_fit2", DummyForecaster(raise_at="fit"))
    forecaster3 = ("raise_fit3", DummyForecaster(raise_at="fit"))
    forecaster4 = ("raise_predict4", DummyForecaster(raise_at="predict"))
    forecaster5 = ("forecaster5", PolynomialTrendForecaster())
    forecaster = FallbackForecaster(
        [forecaster1, forecaster2, forecaster3, forecaster4, forecaster5]
    )
    forecaster.fit(y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()
    y_name_actual = forecaster.current_name
    y_pred_expected = forecaster5[1].predict()
    y_name_expected = forecaster5[0]
    pd.testing.assert_series_equal(y_pred_actual, y_pred_expected)
    assert y_name_actual == y_name_expected
