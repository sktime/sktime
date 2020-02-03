#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = ["DummyForecaster"]
__author__ = "Markus Löning"

from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import _BaseForecasterOptionalFHinFit, DEFAULT_ALPHA
from sktime.utils.validation.forecasting import validate_y, validate_sp, validate_window_length


class DummyForecaster(_BaseForecasterOptionalFHinFit):
    """
    NaiveForecaster is a forecaster that makes forecasts using simple strategies.

    Parameters
    ----------
    strategy : str, {"last", "mean", "seasonal_last"}, optional (default="last")
        Strategy used to make forecasts:

        * "last": forecast the last value in the training series
        * "mean": forecast the mean of (a given window) of the training series

    window_length : int or None, optional (default=None)
        Window length to use in the `mean` strategy. If None, entire training
            series will be used.
    """

    def __init__(self, strategy="last", window_length=None, sp=None):

        # input checks
        # allowed strategies to include: last, constant, seasonal-last, mean, median
        allowed_strategies = ("last", "mean", "seasonal_last")
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy: {strategy}; expected one of {allowed_strategies}")
        self.strategy = strategy
        self.window_length = window_length
        self.sp = sp

        if self.strategy == "last" or self.strategy == "seasonal_last":
            if window_length is not None:
                warn("For the `last` and `seasonal_last` strategy, "
                     "the `window_length` value will be ignored.")

        if self.strategy == "last" or self.strategy == "mean":
            if sp is not None:
                warn("For the `last` and `mean` strategy, "
                     "the `sp` value will be ignored.")

        if self.strategy == "last":
            self._window_length = 1

        if self.strategy == "seasonal_last":
            if sp is None:
                raise NotImplementedError("Estimation of the seasonal periodicity `sp` "
                                          "from the data is not implemented yet; "
                                          "please specify the `sp` value.")
            sp = validate_sp(sp)
            self._window_length = sp
            self._sp = sp

        if self.strategy == "mean":
            self._window_length = validate_window_length(window_length)

        self._last_window = None
        super(DummyForecaster, self).__init__()

    def fit(self, y, fh=None, X=None):

        # input checks
        # in-sample forecast
        if isinstance(fh, str) and fh == "insample":
            raise NotImplementedError()

        # ignore exogenous variables X
        y = validate_y(y)
        self._set_fh(fh)

        # update observation horizon
        self._set_obs_horizon(y.index)

        # set default window length for the mean strategy
        if self.strategy == "mean" and self._window_length is None:
            self._window_length = len(y)

        if self._window_length > len(self._obs_horizon):
            param = "sp" if self.strategy == "seasonal_last" else "window_length"
            raise ValueError(f"The {param}: {self._window_length} is larger than "
                             f"the training series.")

        self._last_window = y.iloc[-self._window_length:]
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):

        # input checks
        self._check_is_fitted()

        if isinstance(fh, str) and fh == "insample":
            raise NotImplementedError()
        if return_pred_int:
            raise NotImplementedError()

        # set fh
        self._set_fh(fh)

        # compute prediction
        if self.strategy == "last":
            y_pred = np.repeat(self._last_window.iloc[-1], len(self.fh))

        if self.strategy == "seasonal_last":
            # we need to replicate the last window if max(fh) is larger than sp,
            # so that we still make forecasts by repeating the last value for that season,
            # assume fh is sorted, i.e. max(fh) == fh[-1]
            if self.fh[-1] > self.sp:
                reps = np.int(np.ceil(self.fh[-1] / self.sp))
                last_window = np.tile(self._last_window, reps=reps)
            else:
                last_window = self._last_window.values

            fh_idx = self.fh - np.min(self.fh)
            y_pred = last_window[fh_idx]

        if self.strategy == "mean":
            y_pred = np.repeat(self._last_window.mean(), len(self.fh))

        # return as series with correct time index
        index = self._now + self.fh
        return pd.Series(y_pred, index=index)

    def update(self, y_new, X_new=None, update_params=False):

        # input checks
        self._check_is_fitted()

        y_new = validate_y(y_new)

        # update observation horizon
        self._set_obs_horizon(y_new.index)

        # update last window
        self._last_window = y_new.iloc[-self._window_length:]

        return self
