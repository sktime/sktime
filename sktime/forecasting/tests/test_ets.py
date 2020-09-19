__author__ = ["Hongyi Yang"]

from numpy.testing import assert_array_equal
from sktime.forecasting.ets import AutoETS
from sktime.datasets import load_airline
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd


# test results against R implementation on airline dataset
def test_airline():
    y = load_airline().astype('float64')

    # Default condition
    # fit <- ets(AirPassengers, model = "ZZZ")
    # components: "M" "A" "M" "TRUE" (error, trend, season, damped)
    # discrepancy lies in damped (True in R but False in statsmodels)
    fit_result_R = ['mul', 'add', 'mul']

    forecaster = AutoETS(autofitting=True, sp=12, n_jobs=-1)
    forecaster.fit(y)
    fitted_forecaster = forecaster._fitted_forecaster
    fit_result = [fitted_forecaster.error,
                  fitted_forecaster.trend,
                  fitted_forecaster.seasonal]

    assert_array_equal(fit_result_R, fit_result)

    # Allow multiplicative trend
    # fit <- ets(AirPassengers, model = "ZZZ",
    # allow.multiplicative.trend = TRUE)
    # components: "M" "M" "M" "TRUE"
    # discrepancy lies in damped (True in R but False in statsmodels)
    fit_result_R = ['mul', 'mul', 'mul']

    forecaster = AutoETS(autofitting=True, sp=12, n_jobs=-1,
                         allow_multiplicative_trend=True)
    forecaster.fit(y)
    fitted_forecaster = forecaster._fitted_forecaster
    fit_result = [fitted_forecaster.error,
                  fitted_forecaster.trend,
                  fitted_forecaster.seasonal]

    assert_array_equal(fit_result_R, fit_result)


# test results against R implementation on gas dataset (forecast package)
def test_gas():
    importr('forecast')
    pandas2ri.activate()
    y = pd.Series(r['gas']).astype('float64')

    # Default condition
    # fit <- ets(gas, model = 'ZZZ')
    # components: "M" "A" "M" "TRUE" (error, trend, season, damped)
    # discrepancy lies in damped (True in R but False in statsmodels)
    fit_result_R = ['mul', 'add', 'mul']

    forecaster = AutoETS(autofitting=True, sp=12, n_jobs=-1)
    forecaster.fit(y)
    fitted_forecaster = forecaster._fitted_forecaster
    fit_result = [fitted_forecaster.error,
                  fitted_forecaster.trend,
                  fitted_forecaster.seasonal]

    assert_array_equal(fit_result_R, fit_result)

    # Allow multiplicative trend
    # fit <- ets(gas, model = "ZZZ",
    # allow.multiplicative.trend = TRUE)
    # components: "M" "M" "M" "TRUE"
    # discrepancy lies in damped (True in R but False in statsmodels)
    fit_result_R = ['mul', 'mul', 'mul']

    forecaster = AutoETS(autofitting=True, sp=12, n_jobs=-1,
                         allow_multiplicative_trend=True)
    forecaster.fit(y)
    fitted_forecaster = forecaster._fitted_forecaster
    fit_result = [fitted_forecaster.error,
                  fitted_forecaster.trend,
                  fitted_forecaster.seasonal]

    assert_array_equal(fit_result_R, fit_result)
