# -*- coding: utf-8 -*-
__author__ = ["Hongyi Yang"]

from numpy.testing import assert_array_equal
from sktime.forecasting.ets import AutoETS
from sktime.datasets import load_airline


# test results against R implementation on airline dataset
y = load_airline()


# Default condition
# fit <- ets(AirPassengers, model = "ZZZ")
# components: "M" "A" "M" "TRUE" (error, trend, season, damped)
# discrepancy lies in damped (True in R but False in statsmodels)
def test_airline_default():
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


# Allow multiplicative trend
# fit <- ets(AirPassengers, model = "ZZZ",
# allow.multiplicative.trend = TRUE)
# components: "M" "M" "M" "TRUE"
# discrepancy lies in damped (True in R but False in statsmodels)
# Test failed on linux environment, fixed by fixing pandas==1.1.5 in #581
# @pytest.mark.skipif(
#     sys.platform == "linux",
#     reason="Skip test due to unknown error on Linux with Python 3.7 and 3.8",
# )
def test_airline_allow_multiplicative_trend():
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
