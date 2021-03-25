# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin


DEFAULT_ALPHA = 0.05


class Croston(_OptionalForecastingHorizonMixin, _SktimeForecaster):
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
            Forecasted demand (on average per period) diff."""

    def __init__(self, alpha=DEFAULT_ALPHA):
        # hyperparameter
        self.alpha = alpha

        # training data
        self._y = None
        self._X = None

        # forecasting horizon
        self._fh = None
        # self.cutoff = None  # reference point for relative fh

        # set _is_fitted to False
        self._is_fitted = False
        super(Croston, self).__init__()

    def fit(self, y, X=None, fh=None):
        if X is not None:
            raise NotImplementedError(
                "Support for exogenous variables is not yet implemented"
            )
        self._set_y_X(y, X)
        self._set_fh(fh)
        self._is_fitted = True

        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int or X is not None:
            raise NotImplementedError()
        self.check_is_fitted()
        self._set_fh(fh)

        future_periods = len(self._fh.to_numpy())
        alpha = self.alpha
        d = self._y.to_numpy()  # Transform the input into a numpy array

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

        index = self.fh.to_absolute(self._cutoff)

        return pd.Series(np.array(f[cols:]), index=index)
