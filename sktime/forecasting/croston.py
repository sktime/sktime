# -*- coding: utf-8 -*-
"""
Implementation of Croston's Method
----------------------------------
Useful for Forecasting Intermittent Demand Time Series.
The Croston() function produces forecasts using Croston’s method.
It simply uses α = 0.1 by default,
and p = 0 is set to be equal to the first observation in each of the series.
This is consistent with the way Croston envisaged the method being used.

Parameters:
-----------
    demand: array-like
        Historical data
    future_periods: int
        Time period for which predictions are required
    alpha: float, optional(default=0.1)
        Smoothing parameter

Returns:
--------
    forecast: array-like
        Forecasted demand (on average per period) diff
"""

import numpy as np
from sktime.forecasting.base._base import BaseForecaster as _BaseForecaster
from sklearn.base import BaseEstimator as _BaseEstimator

DEFAULT_ALPHA = 0.05


class Croston(_BaseForecaster, _BaseEstimator):
    def __init__(self, alpha=DEFAULT_ALPHA):
        # hyperparameter
        self.alpha = alpha

        # training data
        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        self.cutoff = None  # reference point for relative fh

        # set _is_fitted to False
        self._is_fitted = False
        super(_BaseForecaster, self).__init__()

    def fit(self, y, X=None, fh=None):
        self._is_fitted = True
        self._y = y
        self._X = X
        self._fh = fh
        self.cutoff = y.index[-1]
        return self

    def predict(self, fh=None, X=None):
        self._fh = fh
        alpha = self.alpha
        d = np.array(self._y)  # Transform the input into a numpy array
        future_periods = self._fh

        cols = len(d)  # Historical period: i.e the demand array's length
        d = np.append(
            d, [np.nan] * future_periods
        )  # Append np.nan into the demand array to cover future periods

        # level(a), periodicity(p) and forecast(f)
        q, a, f = np.full((3, cols + future_periods), np.nan)
        p = 1  # periods since last demand observation

        # Initialization:
        first_occurrence = np.argmax(d[:cols] > 0)
        q[0] = d[first_occurrence]
        a[0] = 1 + first_occurrence
        f[0] = q[0] / a[0]

        # Create t+1 forecasts:
        for t in range(0, cols):
            if d[t] > 0:
                q[t + 1] = alpha * d[t] + (1 - alpha) * q[t]
                a[t + 1] = alpha * p + (1 - alpha) * a[t]
                f[t + 1] = q[t + 1] / a[t + 1]
                p = 1
            else:
                q[t + 1] = q[t]
                a[t + 1] = a[t]
                f[t + 1] = f[t]
                p += 1

        # Future forecasts:
        q[cols + 1 : cols + future_periods] = q[cols]
        a[cols + 1 : cols + future_periods] = a[cols]
        f[cols + 1 : cols + future_periods] = f[cols]

        return np.array(f[cols:])

    def update(self, y, X=None, update_params=True):
        """Update fitted parameters

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        if update_params is True:
            self._y = y
            self._X = X
            self.cutoff = y.index[-1]

        elif update_params is False:
            self.cutoff = y.index[-1]
        return self
