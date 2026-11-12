#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Unit tests for the FallbackForecaster functionality.

Tests cover the basic operations of the FallbackForecaster, ensuring proper
functionality of fitting, predicting, updating, and handling of errors in the
forecasting process.
"""

__author__ = ["ninedigits"]

import numpy as np
import pandas as pd
import pytest

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.compose import EnsembleForecaster, FallbackForecaster
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils.estimators import MockUnivariateForecasterLogger


class ForecastingError(Exception):
    """Exception raised for errors in the forecasting process.

    Attributes
    ----------
    message : str
        Explanation of the error.

    Methods
    -------
    __init__(message)
    Constructs the ForecastingError with the provided message.
    """

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
    raise_at : str or NoneType
        Stage at which the forecaster should fail. Options are "fit", "predict",
        "update", and None.
    predict_nans : bool
        Replaces the first prediction with a null value when true.

    Raises
    ------
    AttributeError
        If `raise_at` is not one of the valid options.
    """

    def __init__(self, raise_at="fit", predict_nans=False):
        super().__init__()
        __valid__ = ["fit", "predict", "update", None]
        if raise_at not in __valid__:
            raise AttributeError(f"`raise_at` must choose from {__valid__}")
        self.forecaster = NaiveForecaster()
        self.raise_at = raise_at
        self.predict_nans = predict_nans
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
        y_pred = self.forecaster.predict(fh, X)
        if self.predict_nans:
            y_pred.iloc[0] = pd.NA
        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update the forecaster. Optionally fail here."""
        if self.raise_at == "update":
            raise ForecastingError("Intentional failure in update.")
        self.forecaster.update(y, X, update_params)
        return self


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_raises_at_fit():
    """Test dummy forecaster raises at fit"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster = DummyForecaster(raise_at="fit")
    with pytest.raises(ForecastingError):
        forecaster.fit(y=y, fh=[1, 2, 3])


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_raises_at_predict():
    """Test dummy forecaster raises at predict"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster = DummyForecaster(raise_at="predict")
    forecaster.fit(y=y, fh=[1, 2, 3])
    with pytest.raises(ForecastingError):
        forecaster.predict()


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_raises_at_update():
    """Test dummy forecaster raises at update"""
    # Start with negative time series, Theta model will fail here
    y = make_forecasting_problem(random_state=42)
    forecaster = DummyForecaster(raise_at="update")
    forecaster.fit(y=y, fh=[1, 2, 3])
    forecaster.predict()
    with pytest.raises(ForecastingError):
        forecaster.update(y)


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_predicts_nans():
    """Test dummy forecaster predict nans"""
    y = make_forecasting_problem(random_state=42)
    forecaster = DummyForecaster(raise_at=None, predict_nans=True)
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred = forecaster.predict()
    assert y_pred.isna().sum() > 0


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_fails_at_fit():
    """Test FallbackForecaster fails at first fit, second forecaster succeeds"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("fail_fit", DummyForecaster(raise_at="fit")),
        ]
    )
    forecaster2 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("naive", NaiveForecaster()),
        ]
    )
    forecaster = FallbackForecaster(
        [
            (
                "ensemble1_fails_at_fit",
                forecaster1,
            ),
            (
                "ensemble2",
                forecaster2,
            ),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()

    forecaster2.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster2.predict()

    # Assert that the first valid forecaster is trained
    name = forecaster.current_name_
    assert name == "ensemble2"

    # Assert that the first valid forecaster produces the same results as it would
    # on its own
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)

    # Count the number of exceptions raised
    exceptions_raised = forecaster.exceptions_raised_
    assert len(exceptions_raised) == 1


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_fails_at_predict():
    """Test FallbackForecaster fails at predict, second forecaster succeeds"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("raise_at_predict", DummyForecaster(raise_at="predict")),
        ]
    )
    forecaster2 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("naive", NaiveForecaster()),
        ]
    )

    forecaster = FallbackForecaster(
        [
            ("forecaster1_fails_predict", forecaster1),
            ("forecaster2_succeeded", forecaster2),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])

    # Assert predictions line up with the correct forecaster
    y_pred_actual = forecaster.predict()

    forecaster2.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster2.predict()

    # Assert correct forecaster name
    name = forecaster.current_name_
    assert name == "forecaster2_succeeded"

    # Assert correct y_pred
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_fails_twice():
    """First two FallbackForecasters fail, third succeeds"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("fails", DummyForecaster(raise_at="fit")),
        ]
    )
    forecaster2 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("fails", DummyForecaster(raise_at="predict")),
        ]
    )
    forecaster3 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("naive", NaiveForecaster()),
        ]
    )
    forecaster4 = PolynomialTrendForecaster()

    forecaster = FallbackForecaster(
        [
            ("forecaster1_fails_fit", forecaster1),
            ("forecaster2_fails_prd", forecaster2),
            ("forecaster3_succeeded", forecaster3),
            ("forecaster4_notcalled", forecaster4),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()

    forecaster3.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster3.predict()

    # Assert correct forecaster name
    name = forecaster.current_name_
    assert name == "forecaster3_succeeded"

    # Assert correct y_pred
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)

    # Assert correct number of expected exceptions
    exceptions_raised = forecaster.exceptions_raised_
    assert len(exceptions_raised) == 2

    # Assert the correct forecasters failed
    names_raised_actual = [
        vals["forecaster_name"] for vals in exceptions_raised.values()
    ]
    names_raised_expected = ["forecaster1_fails_fit", "forecaster2_fails_prd"]
    assert names_raised_actual == names_raised_expected


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_fails_fit_twice():
    """First two FallbackForecasters fail at fit step, third forecaster succeeds"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("fails", DummyForecaster(raise_at="fit")),
        ]
    )
    forecaster2 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("fails", DummyForecaster(raise_at="fit")),
        ]
    )
    forecaster3 = EnsembleForecaster(
        [
            ("trend", PolynomialTrendForecaster()),
            ("naive", NaiveForecaster()),
        ]
    )
    forecaster4 = PolynomialTrendForecaster()

    forecaster = FallbackForecaster(
        [
            ("forecaster1_fails_fit", forecaster1),
            ("forecaster2_fails_fit", forecaster2),
            ("forecaster3_succeeded", forecaster3),
            ("forecaster4_notcalled", forecaster4),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])

    # Assert predictions line up with the correct forecaster
    y_pred_actual = forecaster.predict()

    forecaster3.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster3.predict()

    # Assert correct forecaster name
    name = forecaster.current_name_
    assert name == "forecaster3_succeeded"

    # Assert correct y_pred
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)

    # Assert correct number of expected exceptions
    exceptions_raised = forecaster.exceptions_raised_
    assert len(exceptions_raised) == 2

    # Assert the correct forecasters failed
    names_raised_actual = [
        vals["forecaster_name"] for vals in exceptions_raised.values()
    ]
    names_raised_expected = ["forecaster1_fails_fit", "forecaster2_fails_fit"]
    assert names_raised_actual == names_raised_expected


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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
    y_name_actual = forecaster.current_name_
    expected_forecaster = forecaster5[1]
    expected_forecaster.fit(y, fh=[1, 2, 3])
    y_pred_expected = expected_forecaster.predict()
    y_name_expected = forecaster5[0]
    pd.testing.assert_series_equal(y_pred_actual, y_pred_expected)
    assert y_name_actual == y_name_expected

    # Assert correct forecaster name
    name = forecaster.current_name_
    assert name == "forecaster5"

    # Assert correct y_pred
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)

    # Assert correct number of expected exceptions
    exceptions_raised = forecaster.exceptions_raised_
    assert len(exceptions_raised) == 4

    # Assert the correct forecasters failed
    names_raised_actual = [
        vals["forecaster_name"] for vals in exceptions_raised.values()
    ]
    names_raised_expected = [
        "raise_predict1",
        "raise_fit2",
        "raise_fit3",
        "raise_predict4",
    ]
    assert names_raised_actual == names_raised_expected


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_fails_twice_simple():
    """First two FallbackForecasters fail, third succeeds"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = DummyForecaster(raise_at="fit")
    forecaster2 = DummyForecaster(raise_at="predict")
    forecaster3 = PolynomialTrendForecaster()
    forecaster4 = NaiveForecaster()

    forecaster = FallbackForecaster(
        [
            ("forecaster1_fails_fit", forecaster1),
            ("forecaster2_fails_prd", forecaster2),
            ("forecaster3_succeeded", forecaster3),
            ("forecaster4_notcalled", forecaster4),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()

    forecaster3.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster3.predict()

    # Assert correct forecaster name
    name = forecaster.current_name_
    assert name == "forecaster3_succeeded"

    # Assert correct y_pred
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)

    # Assert correct number of expected exceptions
    exceptions_raised = forecaster.exceptions_raised_
    assert len(exceptions_raised) == 2

    # Assert the correct forecasters failed
    names_raised_actual = [
        vals["forecaster_name"] for vals in exceptions_raised.values()
    ]
    names_raised_expected = ["forecaster1_fails_fit", "forecaster2_fails_prd"]
    assert names_raised_actual == names_raised_expected


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_fails_many_simple():
    """First two FallbackForecasters fail, third succeeds"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = DummyForecaster(raise_at="predict")
    forecaster2 = DummyForecaster(raise_at="predict")
    forecaster3 = DummyForecaster(raise_at="fit")
    forecaster4 = DummyForecaster(raise_at="fit")
    forecaster5 = DummyForecaster(raise_at="fit")
    forecaster6 = DummyForecaster(raise_at="predict")
    forecaster7 = DummyForecaster(raise_at="fit")
    forecaster8 = DummyForecaster(raise_at="predict")
    forecaster9 = PolynomialTrendForecaster()
    forecaster10 = NaiveForecaster()

    forecaster = FallbackForecaster(
        [
            ("f1", forecaster1),
            ("f2", forecaster2),
            ("f3", forecaster3),
            ("f4", forecaster4),
            ("f5", forecaster5),
            ("f6", forecaster6),
            ("f7", forecaster7),
            ("f8", forecaster8),
            ("target", forecaster9),
            ("notcalled", forecaster10),
        ]
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()

    forecaster9.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster9.predict()

    # Assert correct forecaster name
    name = forecaster.current_name_
    assert name == "target"

    # Assert correct y_pred
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)

    # Assert correct number of expected exceptions
    exceptions_raised = forecaster.exceptions_raised_
    assert len(exceptions_raised) == 8

    # Assert the correct forecasters failed
    names_raised_actual = [
        vals["forecaster_name"] for vals in exceptions_raised.values()
    ]
    names_raised_expected = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
    assert names_raised_actual == names_raised_expected


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_pred_int():
    """Predict interval works bc all forecasters have them enabled, first forecaster
    expected
    """
    y = make_forecasting_problem(random_state=42)
    forecaster1 = NaiveForecaster("mean")
    forecaster2 = NaiveForecaster("last")
    forecaster = FallbackForecaster(
        [("naive_mean", forecaster1), ("naive_last", forecaster2)]
    )
    fh = [1, 2, 3]
    forecaster.fit(y, fh=fh)
    pred_int_actual = forecaster.predict_interval()

    forecaster1.fit(y, fh=fh)
    pred_int_expected = forecaster1.predict_interval()
    pd.testing.assert_frame_equal(pred_int_expected, pred_int_actual)


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_pred_int_raises():
    """Predict int raises because EnsembleForecaster does not have this capability"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = NaiveForecaster("mean")
    forecaster2 = EnsembleForecaster(
        [("naive_last", NaiveForecaster("last")), ("poly", PolynomialTrendForecaster())]
    )
    forecaster = FallbackForecaster(
        [("naive_mean", forecaster1), ("ensemble", forecaster2)]
    )
    fh = [1, 2, 3]
    forecaster.fit(y, fh=fh)
    with pytest.raises(NotImplementedError):
        forecaster.predict_interval()


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_predict_nan_allow():
    """Test FallbackForecaster allow nans in predict"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = DummyForecaster(raise_at="fit")
    forecaster2 = DummyForecaster(raise_at=None, predict_nans=True)
    forecaster = FallbackForecaster(
        [
            ("forecaster1_fails_fit", forecaster1),
            ("forecaster2_pred_nans", forecaster2),
        ],
        nan_predict_policy="ignore",
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()

    forecaster2.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster2.predict()

    assert y_pred_actual.isna().sum() > 0

    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_predict_nan():
    """Test FallbackForecaster raise if nans in predict"""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = DummyForecaster(raise_at="fit")
    forecaster2 = DummyForecaster(raise_at=None, predict_nans=True)
    forecaster3 = PolynomialTrendForecaster()
    forecaster4 = NaiveForecaster()

    forecaster = FallbackForecaster(
        [
            ("forecaster1_fails_fit", forecaster1),
            ("forecaster2_pred_nans", forecaster2),
            ("forecaster3_succeeded", forecaster3),
            ("forecaster4_notcalled", forecaster4),
        ],
        nan_predict_policy="raise",
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    y_pred_actual = forecaster.predict()

    forecaster3.fit(y=y, fh=[1, 2, 3])
    y_pred_expected = forecaster3.predict()

    # Assert correct forecaster name
    name = forecaster.current_name_
    assert name == "forecaster3_succeeded"

    # Assert correct y_pred
    pd.testing.assert_series_equal(y_pred_expected, y_pred_actual)

    # Assert correct number of expected exceptions
    exceptions_raised = forecaster.exceptions_raised_
    assert len(exceptions_raised) == 2

    # Assert the correct forecasters failed
    names_raised_actual = [
        vals["forecaster_name"] for vals in exceptions_raised.values()
    ]
    names_raised_expected = ["forecaster1_fails_fit", "forecaster2_pred_nans"]
    assert names_raised_actual == names_raised_expected


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_warns():
    y = make_forecasting_problem(random_state=42)
    forecaster1 = DummyForecaster(raise_at="fit")
    forecaster2 = DummyForecaster(raise_at=None, predict_nans=True)
    forecaster = FallbackForecaster(
        [
            ("forecaster1_fails_fit", forecaster1),
            ("forecaster2_pred_nans", forecaster2),
        ],
        nan_predict_policy="warn",
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    with pytest.warns(UserWarning):
        forecaster.predict()


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_fallbackforecaster_raises():
    """All forecasters fail when last forecaster predicts nan, raises RuntimeError."""
    y = make_forecasting_problem(random_state=42)
    forecaster1 = DummyForecaster(raise_at="fit")
    forecaster2 = DummyForecaster(raise_at=None, predict_nans=True)
    forecaster = FallbackForecaster(
        [
            ("forecaster1_fails_fit", forecaster1),
            ("forecaster2_pred_nans", forecaster2),
        ],
        nan_predict_policy="raise",
    )
    forecaster.fit(y=y, fh=[1, 2, 3])
    with pytest.raises(RuntimeError):
        forecaster.predict()


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_forecastbylevel_nan_predict():
    from sktime.forecasting.compose import ForecastByLevel
    from sktime.utils._testing.hierarchical import _make_hierarchical

    df = _make_hierarchical(
        hierarchy_levels=(2, 2, 2),
        max_timepoints=10,
        min_timepoints=10,
        same_cutoff=True,
        n_columns=1,
        all_positive=True,
        index_type="period",
        random_state=0,
        add_nan=False,
    )
    forecaster1 = DummyForecaster(raise_at=None, predict_nans=True)
    forecaster2 = NaiveForecaster()
    forecaster = ForecastByLevel(
        FallbackForecaster(
            [
                ("forecaster1_pred_nans", forecaster1),
                ("forecaster2_expected_", forecaster2),
            ],
            nan_predict_policy="raise",
        )
    )
    fh = [1, 2, 3]
    forecaster.fit(y=df, fh=fh)
    y_pred_actual = forecaster.predict()

    forecaster2.fit(y=df, fh=[1, 2, 3])
    y_pred_expected = forecaster2.predict()

    pd.testing.assert_frame_equal(y_pred_expected, y_pred_actual)


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("index_type", ["integer", "datetime", "period"])
@pytest.mark.parametrize("absolute_fh", [False, True])
@pytest.mark.parametrize("remember_data", [False, True])
@pytest.mark.parametrize("update_params", [False, True])
@pytest.mark.parametrize("method", ["update", "update_predict_single"])
def test_fallback_after_update_uses_current_history(
    index_type, absolute_fh, remember_data, update_params, method
):
    """A backup must forecast from the updated observations and cutoff, #11807."""
    if index_type == "integer":
        index = pd.RangeIndex(6)
    elif index_type == "datetime":
        index = pd.date_range("2020-01-01", periods=6, freq="D")
    else:
        index = pd.period_range("2020-01", periods=6, freq="M")
    y = pd.Series([1.0, 2.0, 3.0, 100.0, np.nan], index=index[:5], name="y")
    forecaster = FallbackForecaster(
        [
            ("primary", NaiveForecaster(strategy="mean", window_length=1)),
            ("backup", NaiveForecaster(strategy="last")),
        ],
        nan_predict_policy="raise",
    ).set_config(remember_data=remember_data)
    forecaster.fit(y.iloc[:3], fh=[1])
    fh = ForecastingHorizon(index[5:], is_relative=False) if absolute_fh else [1]

    if method == "update":
        forecaster.update(y.iloc[3:], update_params=update_params)
        actual = forecaster.predict(fh=fh)
    else:
        actual = forecaster.update_predict_single(
            y.iloc[3:], fh=fh, update_params=update_params
        )

    expected = NaiveForecaster(strategy="last").fit(y).predict(fh=fh)
    pd.testing.assert_series_equal(actual, expected)
    assert forecaster.current_name_ == "backup"
    assert forecaster.current_forecaster_.cutoff.equals(forecaster.cutoff)


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("remember_data", [False, True])
def test_fallback_after_update_preserves_exogenous_history(remember_data):
    """Later candidates see all batches of y and X after intervening failures."""
    y = pd.Series(np.arange(7, dtype=float), name="y")
    X = pd.DataFrame({"x": np.arange(8, dtype=float)})
    forecaster = FallbackForecaster(
        [
            ("primary", DummyForecaster(raise_at="predict")),
            ("fit_failure", DummyForecaster(raise_at="fit")),
            ("predict_failure", DummyForecaster(raise_at="predict")),
            ("backup", MockUnivariateForecasterLogger()),
        ]
    ).set_config(remember_data=remember_data)
    forecaster.fit(y.iloc[:3], X=X.iloc[:3], fh=[1])
    forecaster.update(y.iloc[3:5], X=X.iloc[3:5])
    forecaster.update(y.iloc[5:], X=X.iloc[5:7])
    actual = forecaster.predict(X=X.iloc[7:])

    assert forecaster.current_name_ == "backup"
    fit_method, fit_args = forecaster.current_forecaster_.log[0]
    assert fit_method == "_fit"
    pd.testing.assert_series_equal(fit_args["y"], y)
    pd.testing.assert_frame_equal(fit_args["X"], X.iloc[:7])
    pd.testing.assert_series_equal(actual, pd.Series([10], index=[7]))
    assert [
        failure["forecaster_name"] for failure in forecaster.exceptions_raised_.values()
    ] == ["primary", "fit_failure", "predict_failure"]


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("update_params", [False, True])
@pytest.mark.parametrize("method", ["update", "update_predict_single"])
def test_fallback_after_update_forwards_active_update(update_params, method):
    """Updating the history must still update the active candidate exactly once."""
    y = pd.Series(np.arange(5, dtype=float), name="y")
    X = pd.DataFrame({"x": np.arange(6, dtype=float)})
    forecaster = FallbackForecaster([MockUnivariateForecasterLogger()]).fit(
        y.iloc[:3], X=X.iloc[:3], fh=[1]
    )
    if method == "update":
        forecaster.update(y.iloc[3:], X=X.iloc[3:5], update_params=update_params)
        forecaster.predict(X=X.iloc[5:])
    else:
        forecaster.update_predict_single(
            y.iloc[3:], X=X.iloc[3:], update_params=update_params
        )

    updates = [
        args for name, args in forecaster.current_forecaster_.log if name == "_update"
    ]
    assert len(updates) == 1
    assert updates[0]["update_params"] is update_params
    pd.testing.assert_series_equal(updates[0]["y"], y.iloc[3:])
    pd.testing.assert_frame_equal(
        updates[0]["X"], X.iloc[3:5] if method == "update" else X.iloc[3:]
    )


@pytest.mark.skipif(
    not run_test_for_class(FallbackForecaster),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("update_params", [False, True])
def test_fallback_after_update_switches_again(update_params):
    """Keep history current through updates before and after a model switch."""
    y = pd.Series([1.0, 2.0, 3.0], name="y")
    forecaster = FallbackForecaster(
        [
            ("primary", NaiveForecaster(strategy="mean", window_length=1)),
            ("middle", NaiveForecaster(strategy="mean", window_length=2)),
            ("last", NaiveForecaster(strategy="last")),
        ],
        nan_predict_policy="raise",
    ).fit(y, fh=[1])

    batches = [
        pd.Series([10.0, np.nan], index=[3, 4], name="y"),
        pd.Series([20.0, np.nan, np.nan], index=[5, 6, 7], name="y"),
        pd.Series([30.0], index=[8], name="y"),
    ]
    for batch, expected_name, expected_value in zip(
        batches, ["middle", "last", "last"], [10.0, 20.0, 30.0]
    ):
        forecaster.update(batch, update_params=update_params)
        actual = forecaster.predict()
        pd.testing.assert_series_equal(
            actual, pd.Series([expected_value], index=[batch.index[-1] + 1], name="y")
        )
        assert forecaster.current_name_ == expected_name
