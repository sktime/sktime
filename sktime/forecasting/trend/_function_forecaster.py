# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements FunctionForecaster."""

__author__ = ["benheid"]
__all__ = ["FunctionForecaster"]

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class FunctionForecaster(BaseForecaster):
    """The FunctionForecaster uses a function that is fitted using scipy curve_fit.

    Parameters
    ----------
    function: Callable
        The function that should be fitted and used to make forecasts.
    initial_params : list
        The initial parameters of the functions that should be used for fitting


    Examples
    --------
    >>> from sktime.forecasting.trend import FunctionForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> def linear_function(x, a, b):
    ...     return a * x + b
    >>> forecaster = FunctionForecaster(function=linear_function,
    ...                                 initial_params=[1, 1])
    >>> forecaster.fit(y)
    FunctionForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, function, initial_params):
        self.function = function
        self.initial_params = initial_params
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
            self.function, np.array(t), y.values, self.initial_params
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
        params = {
            "function": _test_function,
            "initial_params": [1, 1],
        }

        return params


def _test_function(x, a, b):
    return a * x + b
