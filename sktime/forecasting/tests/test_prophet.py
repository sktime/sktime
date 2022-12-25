# -*- coding: utf-8 -*-
"""Tests for Prophet.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["fkiraly"]

import pandas as pd
import pytest

from sktime.forecasting.fbprophet import Prophet
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("prophet", severity="none"),
    reason="skip test if required soft dependency not available",
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
        assert y_pred.index.is_integer()
    if indextype == "period":
        assert isinstance(y_pred.index, pd.PeriodIndex)


@pytest.mark.skipif(
    not _check_soft_dependencies("prophet", severity="none"),
    reason="skip test if required soft dependency not available",
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
