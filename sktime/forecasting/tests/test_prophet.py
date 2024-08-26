"""Tests for Prophet.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["fkiraly", "tpvasconcelos"]

from unittest import mock

import pandas as pd
import pytest

from sktime.forecasting.fbprophet import Prophet
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(Prophet),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("indextype", ["range", "period"])
def test_prophet_nonnative_index(indextype):
    """Check prophet with RangeIndex and PeriodIndex."""
    y = pd.DataFrame({"a": [1, 2, 3, 4]})
    X = pd.DataFrame({"b": [1, 5, 3, 3, 5, 6], "c": [5, 5, 3, 3, 4, 2]})

    if indextype == "period":
        y.index = pd.period_range("2000-01-01", periods=4)
        X.index = pd.period_range("2000-01-01", periods=6)

    X_train = X.iloc[:4]
    X_test = X.iloc[4:]

    fh = [1, 2]

    f = Prophet()
    f.fit(y, X=X_train)
    y_pred = f.predict(fh=fh, X=X_test)

    if indextype == "range":
        assert pd.api.types.is_integer_dtype(y_pred.index)
    if indextype == "period":
        assert isinstance(y_pred.index, pd.PeriodIndex)


@pytest.mark.skipif(
    not run_test_for_class(Prophet),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("convert_to_datetime", [False, True])
def test_prophet_period_fh(convert_to_datetime):
    """Test Prophet with PeriodIndex based forecasting horizon, see issue #3537."""
    from sktime.datasets import load_airline
    from sktime.forecasting.base import ForecastingHorizon

    y = load_airline()

    if convert_to_datetime:
        y = y.to_timestamp(freq="M")

    fh_index = pd.PeriodIndex(pd.date_range("1961-01", periods=36, freq="M"))
    fh = ForecastingHorizon(fh_index, is_relative=False)

    forecaster = Prophet(
        seasonality_mode="multiplicative",
        n_changepoints=int(len(y) / 12),
        add_country_holidays={"country_name": "UnitedStates"},
        yearly_seasonality=True,
    )

    forecaster.fit(y)

    y_pred = forecaster.predict(fh)

    assert len(y_pred) == len(fh_index)
    if convert_to_datetime:
        assert isinstance(y_pred.index, pd.DatetimeIndex)
        assert (y_pred.index == fh_index.to_timestamp()).all()
    else:
        assert isinstance(y_pred.index, pd.PeriodIndex)
        assert (y_pred.index == fh_index).all()


@pytest.mark.skipif(
    not run_test_for_class(Prophet),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "fit_kwargs", [None, {"foo": "bar"}, {"foo1": "bar1", "foo2": "bar2"}]
)
def test_prophet_fit_kwargs_are_passed_down(fit_kwargs: dict):
    """Test that the `fit_kwargs` hyperparameter is passed down to Prophet.fit()."""
    from sktime.datasets import load_airline
    from sktime.forecasting.fbprophet import Prophet

    y = load_airline()
    with mock.patch("prophet.forecaster.Prophet.fit") as mock_fit:
        forecaster = Prophet(fit_kwargs=fit_kwargs)
        forecaster.fit(y)
        mock_fit.assert_called_once()
        assert mock_fit.call_args.args == ()
        call_kwargs = mock_fit.call_args.kwargs
        # `df` should always be one of the arguments but
        # we don't care about its actual value here
        call_kwargs.pop("df")
        assert call_kwargs == (fit_kwargs or {})


@pytest.mark.skipif(
    not run_test_for_class(Prophet),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_prophet_added_seasonality_is_not_regressor():
    """Test that an single added seasonality is not added as a regressor"""
    import random

    from sktime.datasets import load_airline
    from sktime.forecasting.fbprophet import Prophet

    y = (
        load_airline()
        .to_frame()
        .assign(test_condition=lambda df: random.choices([True, False], k=len(df)))  # noqa: S311
    )

    forecaster = Prophet(
        add_seasonality={
            "name": "testing",
            "period": 7,
            "fourier_order": 20,
            "condition_name": "test_condition",
        },
    )

    forecaster.fit(y=y["Number of airline passengers"], X=y[["test_condition"]])
    assert forecaster.is_fitted


@pytest.mark.skipif(
    not run_test_for_class(Prophet),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_prophet_added_seasonalities_are_not_regressors():
    """Test that multiple added seasonalities are not added as regressors"""
    import random

    from sktime.datasets import load_airline
    from sktime.forecasting.fbprophet import Prophet

    y = (
        load_airline()
        .to_frame()
        .assign(
            test_condition1=lambda df: random.choices([True, False], k=len(df)),  # noqa: S311
            test_condition2=lambda df: random.choices([True, False], k=len(df)),  # noqa: S311
        )
    )

    forecaster = Prophet(
        add_seasonality=[
            {
                "name": "testing1",
                "period": 7,
                "fourier_order": 20,
                "condition_name": "test_condition1",
            },
            {
                "name": "testing2",
                "period": 7,
                "fourier_order": 20,
                "condition_name": "test_condition2",
            },
        ]
    )

    forecaster.fit(
        y=y["Number of airline passengers"], X=y[["test_condition1", "test_condition2"]]
    )
    assert forecaster.is_fitted


@pytest.mark.skipif(
    not run_test_for_class(Prophet),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("constant_timeseries", [True, False])
def test_prophet_fitted_params(constant_timeseries):
    """
    Test if the fitted parameters have the expected number of dimensions.

    For constant timeseries, get_fitted_params was raising unexpeceted error
    (see issue #6982)
    """
    expected_param_ndims = {
        "k": 0,
        "m": 0,
        "sigma_obs": 0,
        "delta": 1,
        "beta": 1,
    }

    if not constant_timeseries:
        from sktime.datasets import load_airline

        y = load_airline()
    else:
        y = pd.DataFrame(
            index=pd.period_range(start="2022-01", periods=18, freq="Q"),
            data={"value": 0},
        )
    forecaster = Prophet()
    forecaster.fit(y)
    fitted_params = forecaster.get_fitted_params()

    # Assert parameters have the expected number of dimensions
    for param, expected_ndim in expected_param_ndims.items():
        assert fitted_params[param].ndim == expected_ndim
