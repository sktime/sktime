# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements CurveFitForecaster."""

__author__ = ["benheid"]
__all__ = ["CurveFitForecaster"]

import pandas as pd
from scipy.optimize import curve_fit

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.trend._util import _get_X_numpy_int_from_pandas


class CurveFitForecaster(BaseForecaster):
    """The CurveFitForecaster takes a function and fits it by using scipy curve_fit.

    The CurveFitForecaster uses the scipy curve_fit method to determine the optimal
    parameters for a given function.

    If the index is an integer index, it directly uses the index values.
    If the index is a `pd.DatetimeIndex` or a `pd.PeriodIndex`, the index values
    are transformed into floats using two distinct approaches:

    1. For a `pd.DatetimeIndex`, it calculates the number of days since 1970-01-01.
       For a `pd.PeriodIndex`, it computes the number of (full) periods since
        1970-01-01.

    2. For a `pd.DatetimeIndex`, it calculates the number of days since the first
       index value.
       For a  `pd.PeriodIndex`, it calculates or the number of (dull) periods since
       the first index value.
    Furthermore, the difference between the index values can be normalised by
    setting the difference between the first and the second index value to one.

    In `fit`
    1. The index of the input time series is transformed to a list of floats.
    2. The scipy curve_fit is called using the list of floats as x values,
       and the time series values as y values.
    In `predict`
    1. The ForecastingHorizon is transformed to a list of floats.
    2. The list of floats is passed together with the fitted parameters to the
       function to provide the forecast.

    Parameters
    ----------
    function: Callable[[Iterable[float], ...], Iterable[float]]
        The function that should be fitted and used to make forecasts.
        The signature of the functions is `function(x, ...)`.
        It takes the independent variables as first argument and the parameters
        to fit as separate remaining arguments.
        See scipy.optimize.curve_fit for more information.
    curve_fit_params: dict, default=None
        Additional parameters that should be passed to the curve_fit method.
        See scipy.optimize.curve_fit for more information.
    origin: {"unix_zero", "first_index"}, default="unix_zero"
        The origin of the time series index.
    normalise_index: bool, default=False
        If True, the differences between the index values are normalised by
        setting the difference between the first and second index value to one.

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
        origin="unix_zero",
        normalise_index=False,
    ):
        self.function = function
        self.curve_fit_params = curve_fit_params
        self._curve_fit_params = (
            curve_fit_params if curve_fit_params is not None else {}
        )
        if origin not in ["unix_zero", "first_index"]:
            raise ValueError(
                f"origin must be 'unix_zero' or 'first_index', but found {origin}"
            )
        self.origin = origin
        self.normalise_index = normalise_index
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
        x = _get_X_numpy_int_from_pandas(y.index)[:, 0]
        start = _get_X_numpy_int_from_pandas(self._y.index[:1])[:, 0]

        if self.origin == "first_index":
            x = x - start
        self.delta_ = None
        if self.normalise_index:
            if len(x) > 1:
                self.delta_ = x[1] - x[0]
                x = x // self.delta_

        self.params_ = curve_fit(self.function, x, y.values, **self._curve_fit_params)
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
        fh = self.fh.to_absolute_index(self.cutoff)
        x = _get_X_numpy_int_from_pandas(fh)[:, 0]
        start = _get_X_numpy_int_from_pandas(self._y.index[:1])[:, 0]

        if self.origin == "first_index":
            x = x - start
        if self.delta_ is not None:
            x = x // self.delta_

        return pd.Series(
            self.function(x, *self.params_[0]),
            index=fh,
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
            "origin": "first_index",
            "normalise_index": True,
        }

        return [params1, params2]


def _test_function(x, a, b):
    return a * x + b


def _test_function_2(x, a, b, c):
    return a * x**2 + b * x + c
