#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["NaiveForecaster"]
__author__ = "Markus Löning"

from warnings import warn

import numpy as np
import pandas as pd
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import BaseLastWindowForecaster
from sktime.forecasting.base._sktime import OptionalForecastingHorizonMixin
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.validation.forecasting import check_window_length


class NaiveForecaster(OptionalForecastingHorizonMixin,
                      BaseLastWindowForecaster):
    """
    NaiveForecaster is a forecaster that makes forecasts using simple
    strategies.

    Parameters
    ----------
    strategy : str{"last", "mean"}, optional (default="last")
        Strategy used to make forecasts:

        * "last" and sp is None: forecast the last value in the
                                training series
        * "mean" and sp is None: forecast the mean of (a given window)
                                of the training series
        * "last" and sp is not None: forecasts the last value of the same
                                season in the training series
        * "mean" and sp is not None: forecast the mean (of a given window)
                                of the same season in the training series

    sp : int or None, optional (default=None)
        Seasonal periodicity to use in the seasonal forecast strategies.
         If None, naive strategy will be used

    window_length : int or None, optional (default=None)
        Window length to use in the `mean` strategy. If None, entire training
            series will be used.
    """

    def __init__(self, strategy="last", window_length=None, sp=None):
        super(NaiveForecaster, self).__init__()
        # input checks
        # allowed strategies to include: last, constant, seasonal-last,
        # mean, median
        allowed_strategies = ("last", "mean")
        if strategy not in allowed_strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}; expected one of "
                f"{allowed_strategies}")
        self.strategy = strategy
        self.sp = sp
        self.window_length = window_length

        self.sp_ = None

    def fit(self, y_train, fh=None, X_train=None):
        """Fit to training data.

        Parameters
        ----------
        y_train : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X_train : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """  # X_train is ignored
        self._set_oh(y_train)
        self._set_fh(fh)

        if self.strategy == "last":
            if self.window_length is not None:
                warn("For the `last` strategy, "
                     "the `window_length_` value will be ignored.")

        if self.strategy == "last" and self.sp is None:
            self.window_length_ = 1

        if self.strategy == "last" and self.sp is not None:
            self.sp_ = check_sp(self.sp)

            # window length we need for forecasts is just the
            # length of seasonal periodicity
            self.window_length_ = self.sp_

        if self.strategy == "mean":
            # check window length is greater than sp for seasonal mean
            if self.window_length is not None and self.sp is not None:
                if self.window_length < self.sp:
                    param1, param2 = "window_length", "sp"
                    raise ValueError(f"The {param1}: {self.window_length}"
                                     f" is lesser than the"
                                     f" {param2}: {self.sp}.")
            self.window_length_ = check_window_length(self.window_length)
            self.sp_ = check_sp(self.sp)

        #  if not given, set default window length for the mean strategy
        if self.strategy == "mean" and self.window_length is None:
            self.window_length_ = len(y_train)

        # check window length
        if self.window_length_ > len(self.oh):
            param = "sp" if self.strategy == "last" and self.sp is not None \
                else "window_length_"
            raise ValueError(
                f"The {param}: {self.window_length_} is larger than "
                f"the training series.")

        self._is_fitted = True
        return self

    def _predict_last_window(self, fh, X=None, return_pred_int=False,
                             alpha=DEFAULT_ALPHA):
        """Internal predict"""
        last_window = self._get_last_window()

        # if last window only contains missing values, return nan
        if np.all(np.isnan(last_window)) or len(last_window) == 0:
            return self._predict_nan(fh)

        elif self.strategy == "last" and self.sp is None:
            return np.repeat(last_window[-1], len(fh))

        elif self.strategy == "last" and self.sp is not None:
            # we need to replicate the last window if max(fh) is larger than
            # sp,
            # so that we still make forecasts by repeating the last value
            # for that season,
            # assume fh is sorted, i.e. max(fh) == fh[-1]
            if fh[-1] > self.sp_:
                reps = np.int(np.ceil(fh[-1] / self.sp_))
                last_window = np.tile(last_window, reps=reps)

            # get zero-based index by subtracting the minimum
            fh_idx = fh.index_like(self.cutoff)
            return last_window[fh_idx]

        elif self.strategy == "mean" and self.sp is None:
            return np.repeat(np.nanmean(last_window), len(fh))

        elif self.strategy == "mean" and self.sp is not None:
            last_window = pd.DataFrame(data=last_window,
                                       columns=['data'])
            # computing last season's mean and imputing it into last season
            for i in range(self.sp_):
                if any(last_window.index % self.sp_ == i) is True:
                    last_window.\
                        at[last_window[last_window.index % self.sp_ == i].
                            index[-1], 'data'] =\
                        last_window[last_window.index % self.sp_ == i].mean()

            # we need to replicate the last window if max(fh) is
            # larger than sp,
            # so that we still make forecasts by repeating the
            # last value for that season,
            # assume fh is sorted, i.e. max(fh) == fh[-1]
            # only slicing all the last seasons into last_window
            if fh[-1] > self.sp_:
                reps = np.int(np.ceil(fh[-1] / self.sp_))
                last_window =\
                    np.tile(last_window['data'].tail(self.sp_).to_numpy(),
                            reps=reps)
            else:
                last_window =\
                    last_window['data'].tail(self.sp_).to_numpy()

            # get zero-based index by subtracting the minimum
            fh_idx = fh.index_like(self.cutoff)
            return last_window[fh_idx]
