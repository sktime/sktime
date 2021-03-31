# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin


DEFAULT_SMOOTHING = 0.05
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
        len_fh: int
            Time period for which predictions are required
        smoothing: float, optional(default=0.1)
            smoothing parameter

    Returns:
    --------
        forecast: array-like
            Forecasted demand (on average per period) diff."""

    def __init__(self, smoothing=DEFAULT_SMOOTHING):
        # hyperparameter
        self.smoothing = smoothing
        self.f = None
        super(Croston, self).__init__()

    def fit(self, y, X=None, fh=None):
        if X is not None:
            raise NotImplementedError(
                "Support for exogenous variables is not yet implemented"
            )
        self._set_y_X(y, X)
        self._set_fh(fh)
        self._is_fitted = True

        n_timepoints = len(y)  # Historical period: i.e the input array's length
        smoothing = self.smoothing

        y = y.to_numpy()  # Transform the input into a numpy array
        # Fit the parameters: level(q), periodicity(a) and forecast(f)
        q, a, f = np.full((3, n_timepoints + 1), np.nan)
        p = 1  # periods since last demand observation

        # Initialization:
        first_occurrence = np.argmax(y[:n_timepoints] > 0)
        q[0] = y[first_occurrence]
        a[0] = 1 + first_occurrence
        f[0] = q[0] / a[0]

        # Create t+1 forecasts:
        for t in range(0, n_timepoints):
            if y[t] > 0:
                q[t + 1] = smoothing * y[t] + (1 - smoothing) * q[t]
                a[t + 1] = smoothing * p + (1 - smoothing) * a[t]
                f[t + 1] = q[t + 1] / a[t + 1]
                p = 1
            else:
                q[t + 1] = q[t]
                a[t + 1] = a[t]
                f[t + 1] = f[t]
                p += 1
        self.f = f

        return self

    def predict(
        self,
        fh=None,
        X=None,
        return_pred_int=False,
        smoothing=DEFAULT_SMOOTHING,
        alpha=DEFAULT_ALPHA,
    ):
        if return_pred_int or X is not None:
            raise NotImplementedError()

        self.check_is_fitted()
        self._set_fh(fh)
        len_fh = len(self._fh.to_numpy())
        f = self.f

        # Predicting future forecasts:
        y_pred = np.full(len_fh, f[-1])

        index = self.fh.to_absolute(self._cutoff)
        return pd.Series(y_pred, index=index)
