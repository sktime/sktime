# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements FunctionForecaster."""

__author__ = ["benheid"]
__all__ = ["CurveFitForecaster"]

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class CurveFitForecaster(BaseForecaster):
    """The CurveFitForecaster takes a function and tits it by using scipy curve_fit.

    The CurveFitForecaster applies the scipy curve_fit method to find the best
    parameter of a function. This function maps a list of integers to real values.
    The integers are derrived from the index of the input time series.

    In `fit`
    1. The index of the input time series is transformed to a list of Integers.
    2. The scipy curve_fit is called using the list of integers as x values,
       and the time series values as y values.
    In `predict`
    1. The ForecastingHorizon is transformed to a list of integers.
    2. The list of integers is passed together with the fitted parameters to the
       function to provide the forecast.


    Parameters
    ----------
    function: Callable
        The function that should be fitted and used to make forecasts.
        The signature of the functions is `function(x, ...)`.
        It takes the independet variables as first argument and the parametrs
        to fit as separate remaining arguments.
        See scipy.optimize.curve_fit for more information.
    curve_fit_params: dict, default=None
        Additional parameters that should be passed to the curve_fit method.
        See scipy.optimize.curve_fit for more information.


    Examples
    --------
    >>> from sktime.forecasting.trend import CurveFitForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> def linear_function(x, a, b):
    ...     return a * x + b
    >>> forecaster = CurveFitForecaster(function=linear_function,
    ...                                 curve_fit_params={"p0":[-1, 1]})
    >>> forecaster.fit(y)
    CurveFitForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(
        self,
        function,
        curve_fit_params=None,
    ):
        self.function = function
        self.curve_fit_params = curve_fit_params
        self._curve_fit_params = (
            curve_fit_params if curve_fit_params is not None else {}
        )
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame
            time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
            Ignored in fit
        X : pd.DataFrame, optional (default=None)
            Ignored in fit

        Returns
        -------
        self : reference to self
        """
        t = ForecastingHorizon(y.index, is_relative=False).to_relative(self.cutoff)
        self.params_ = curve_fit(
            self.function, np.array(t), y.values, **self._curve_fit_params
        )
        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon,
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Ignored in predict

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        t = fh.to_relative(self.cutoff)
        return pd.Series(
            self.function(np.array(t), *self.params_[0]),
            index=list(fh.to_absolute(self.cutoff)),
            name=self._y.name,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class.
        """
        params1 = {
            "function": _test_function,
            "curve_fit_params": {
                "p0": [1, 1],
                "sigma": None,
                "absolute_sigma": True,
                "check_finite": True,
                "bounds": [(-100, -100), (100, 100)],
                "method": "dogbox",
            },
        }

        params2 = {
            "function": _test_function_2,
            "curve_fit_params": {
                "method": "trf",
            },
        }

        return [params1, params2]


def _test_function(x, a, b):
    return a * x + b


def _test_function_2(x, a, b, c):
    return a * x**2 + b * x + c
