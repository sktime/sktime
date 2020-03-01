#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = ["NaiveForecaster"]
__author__ = "Markus Löning"

from warnings import warn

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseLastWindowForecaster, OptionalForecastingHorizonMixin
from sktime.forecasting.base import DEFAULT_ALPHA
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.validation.forecasting import check_window_length


class NaiveForecaster(OptionalForecastingHorizonMixin, BaseLastWindowForecaster):
    """
    NaiveForecaster is a forecaster that makes forecasts using simple strategies.

    Parameters
    ----------
    strategy : str{"last", "mean", "seasonal_last"}, optional (default="last")
        Strategy used to make forecasts:

        * "last": forecast the last value in the training series
        * "mean": forecast the mean of (a given window) of the training series
        * "seasonal_last": forecasts the last value of the same season in the training series

    window_length : int or None, optional (default=None)
        Window length to use in the `mean` strategy. If None, entire training
            series will be used.
    """

    def __init__(self, strategy="last", window_length=None, sp=None):
        super(NaiveForecaster, self).__init__()
        # input checks
        # allowed strategies to include: last, constant, seasonal-last, mean, median
        allowed_strategies = ("last", "mean", "seasonal_last")
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy: {strategy}; expected one of {allowed_strategies}")
        self.strategy = strategy
        self.sp = sp
        self._window_length = window_length

        if strategy in ("last", "seasonal_last"):
            if self.window_length is not None:
                warn("For the `last` and `seasonal_last` strategy, "
                     "the `window_length` value will be ignored.")

        if self.strategy in ("last", "mean"):
            if self.sp is not None:
                warn("For the `last` and `mean` strategy, "
                     "the `sp` value will be ignored.")

        if self.strategy == "last":
            self._window_length = 1

        if self.strategy == "seasonal_last":
            if self.sp is None:
                raise NotImplementedError("Automatic estimation of the seasonal periodicity `sp` "
                                          "from the data is not implemented yet; "
                                          "please specify the `sp` value.")
            self._sp = check_sp(sp)

            # window length we need for forecasts is just the
            # length of seasonal periodicity
            self._window_length = self._sp

        if self.strategy == "mean":
            self._window_length = check_window_length(window_length)

    def fit(self, y_train, fh=None, X_train=None):
        """Fit"""
        # X_train is ignored
        self._set_oh(y_train)
        self._set_fh(fh)

        #  if not given, set default window length for the mean strategy
        if self.strategy == "mean" and self._window_length is None:
            self._window_length = len(y_train)

        # check window length
        if self._window_length > len(self.oh):
            param = "sp" if self.strategy == "seasonal_last" else "window_length"
            raise ValueError(f"The {param}: {self._window_length} is larger than "
                             f"the training series.")

        self._is_fitted = True
        return self

    def _predict_last_window(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Internal predict"""
        # if last window only contains missing values, return nan
        last_window = self._get_last_window()
        if np.all(np.isnan(last_window)):
            return self._predict_nan(fh)

        elif self.strategy == "last":
            return np.repeat(last_window[-1], len(fh))

        elif self.strategy == "seasonal_last":
            # we need to replicate the last window if max(fh) is larger than sp,
            # so that we still make forecasts by repeating the last value for that season,
            # assume fh is sorted, i.e. max(fh) == fh[-1]
            if fh[-1] > self.sp:
                reps = np.int(np.ceil(fh[-1] / self.sp))
                last_window = np.tile(last_window, reps=reps)

            # get zero-based index by subtracting the minimum
            fh_idx = self._get_array_index_fh(fh)
            return last_window[fh_idx]

        elif self.strategy == "mean":
            return np.repeat(np.nanmean(last_window), len(fh))

