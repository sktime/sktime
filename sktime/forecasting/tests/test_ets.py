# -*- coding: utf-8 -*-
"""ETS tests."""

__author__ = ["Hongyi Yang"]

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from sktime.datasets import load_airline
from sktime.forecasting.ets import AutoETS
from sktime.utils.validation._dependencies import _check_soft_dependencies

# test results against R implementation on airline dataset
y = load_airline()

# dummy time series that results in infinite ICs
inf_ic_ts = pd.Series(
    10 * np.sin(np.array(range(0, 264)) / 10) + 12,
    pd.date_range("2017-01-01", periods=264, freq="W"),
)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_airline_default():
    """
    Default condition.

    fit <- ets(AirPassengers, model = "ZZZ")
    components: "M" "A" "M" "TRUE" (error, trend, season, damped)
    discrepancy lies in damped (True in R but False in statsmodels)
    """
    fit_result_R = ["mul", "add", "mul"]

    forecaster = AutoETS(auto=True, sp=12, n_jobs=-1)
    forecaster.fit(y)
    fitted_forecaster = forecaster._fitted_forecaster
    fit_result = [
        fitted_forecaster.error,
        fitted_forecaster.trend,
        fitted_forecaster.seasonal,
    ]

    assert_array_equal(fit_result_R, fit_result)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.xfail(reason="flaky results on linux")
def test_airline_allow_multiplicative_trend():
    """
    Allow multiplicative trend.

    fit <- ets(AirPassengers, model = "ZZZ",
    allow.multiplicative.trend = TRUE)
    components: "M" "M" "M" "TRUE"
    discrepancy lies in damped (True in R but False in statsmodels)
    Test failed on linux environment, fixed by fixing pandas==1.1.5 in #581
    """
    fit_result_R = ["mul", "mul", "mul"]

    forecaster = AutoETS(auto=True, sp=12, n_jobs=-1, allow_multiplicative_trend=True)
    forecaster.fit(y)
    fitted_forecaster = forecaster._fitted_forecaster
    fit_result = [
        fitted_forecaster.error,
        fitted_forecaster.trend,
        fitted_forecaster.seasonal,
    ]

    assert_array_equal(fit_result_R, fit_result)


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
def test_inf_ic_true():
    """Ignore infinite IC models when ignore_inf_ic is `True`."""
    forecaster = AutoETS(auto=True, sp=52, n_jobs=-1, ignore_inf_ic=True)
    forecaster.fit(inf_ic_ts)
    fitted_forecaster = forecaster._fitted_forecaster
    # check that none of the information criteria are infinite
    assert (
        np.isfinite(fitted_forecaster.aic)
        and np.isfinite(fitted_forecaster.bic)
        and np.isfinite(fitted_forecaster.aicc)
    )


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.xfail
def test_inf_ic_false():
    """Don't ignore infinite IC models when ignore_inf_ic is False."""
    forecaster = AutoETS(auto=True, sp=52, n_jobs=-1, ignore_inf_ic=False)
    forecaster.fit(inf_ic_ts)
    fitted_forecaster = forecaster._fitted_forecaster
    # check that all of the information criteria are infinite
    assert (
        np.isinf(fitted_forecaster.aic)
        and np.isinf(fitted_forecaster.bic)
        and np.isinf(fitted_forecaster.aicc)
    )
