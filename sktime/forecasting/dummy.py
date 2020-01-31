#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = ["DummyForecaster"]
__author__ = "Markus Löning"

from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import _BaseForecasterOptionalFHinFit
from sktime.utils.validation.forecasting import validate_y


class DummyForecaster(_BaseForecasterOptionalFHinFit):
    """
    NaiveForecaster is a forecaster that makes forecasts using simple strategies.

    Parameters
    ----------
    strategy : str, {"last", "mean"}, optional (default="last")
        Strategy used to make forecasts:

        * "last": forecast the last value in the training series
        * "mean": forecast the mean of (a given window) of the training series

    window_length : int or None, optional (default=None)
        Window length to use in the `mean` strategy. If None, entire training
            series will be used.
    """

    def __init__(self, strategy="last", window_length=None):

        # input checks
        # allowed strategies to include: last, constant, seasonal-last, mean, median
        allowed_strategies = ("last", "mean")
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy: {strategy}; expected one of {allowed_strategies}")
        self.strategy = strategy
        self.window_length = window_length

        if self.strategy == "last":
            if window_length is not None:
                warn("For the `last` strategy the `window_length` value will be ignored.")
            self._window_length = 1

        if self.strategy == "mean":
            if isinstance(window_length, (int, np.integer)) and window_length < 2:
                raise ValueError("`window_length` must be a positive integer >= 2 or None; "
                                 "for `window_length`=1, use the `last` strategy.")
            self._window_length = window_length

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

        if self.strategy == "mean" and self._window_length is None:
            self._window_length = len(y)

        if self._window_length > len(self._obs_horizon):
            raise ValueError(f"The window length: {self._window_length} is larger than "
                             f"the training series.")

        self._last_window = y.iloc[-self._window_length:]
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=_BaseForecasterOptionalFHinFit._DEFAULT_ALPHA):

        # input checks
        self._check_is_fitted()

        if isinstance(fh, str) and fh == "insample":
            raise NotImplementedError()
        if return_conf_int:
            raise NotImplementedError()

        # set fh
        self._set_fh(fh)

        # compute prediction
        if self.strategy == "last":
            y_pred = self._last_window.iloc[-1]

        if self.strategy == "mean":
            y_pred = self._last_window.mean()

        # return as series with correct time index
        return pd.Series(np.repeat(y_pred, len(self.fh)), index=self._now + self.fh)

    def update(self, y_new, X_new=None, update_params=False):

        # input checks
        self._check_is_fitted()

        y_new = validate_y(y_new)

        # update observation horizon
        self._set_obs_horizon(y_new.index)

        # update last window
        self._last_window = y_new.iloc[-self._window_length:]

        return self
