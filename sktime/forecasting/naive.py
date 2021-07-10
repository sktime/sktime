#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["NaiveForecaster"]
__author__ = ["Markus Löning", "Piyush Gade"]

from warnings import warn

import numpy as np

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.utils.validation.forecasting import check_sp
from sktime.utils.validation import check_window_length


class NaiveForecaster(_BaseWindowForecaster):
    """
    NaiveForecaster is a forecaster that makes forecasts using simple
    strategies.

    Parameters
    ----------
    strategy : str{"last", "mean", "drift"}, optional (default="last")
        Strategy used to make forecasts:

        * "last" : forecast the last value in the
                    training series when sp is 1.
                    When sp is not 1,
                    last value of each season
                    in the last window will be
                    forecasted for each season.
        * "mean" : forecast the mean of last window
                     of training series when sp is 1.
                     When sp is not 1, mean of all values
                     in a season from last window will be
                     forecasted for each season.
        * "drift": forecast by fitting a line between the
                    first and last point of the window and
                     extrapolating it into the future

    sp : int, optional (default=1)
        Seasonal periodicity to use in the seasonal forecasting.

    window_length : int or None, optional (default=None)
        Window length to use in the `mean` strategy. If None, entire training
            series will be used.

    Example
    ----------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> forecaster.fit(y)
    NaiveForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {"requires-fh-in-fit": False}

    def __init__(self, strategy="last", window_length=None, sp=1):
        super(NaiveForecaster, self).__init__()
        self.strategy = strategy
        self.sp = sp
        self.window_length = window_length

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        # X_train is ignored

        n_timepoints = y.shape[0]

        if self.strategy == "last":
            if self.sp == 1:
                if self.window_length is not None:
                    warn(
                        "For the `last` strategy, "
                        "the `window_length` value will be ignored if `sp` "
                        "== 1."
                    )
                self.window_length_ = 1

            else:
                self.sp_ = check_sp(self.sp)

                # window length we need for forecasts is just the
                # length of seasonal periodicity
                self.window_length_ = self.sp_

        elif self.strategy == "mean":
            # check window length is greater than sp for seasonal mean
            if self.window_length is not None and self.sp != 1:
                if self.window_length < self.sp:
                    raise ValueError(
                        f"The `window_length`: "
                        f"{self.window_length} is smaller than "
                        f"`sp`: {self.sp}."
                    )
            self.window_length_ = check_window_length(self.window_length, n_timepoints)
            self.sp_ = check_sp(self.sp)

            #  if not given, set default window length for the mean strategy
            if self.window_length is None:
                self.window_length_ = len(y)

        elif self.strategy == "drift":
            if self.sp != 1:
                warn("For the `drift` strategy, the `sp` value will be ignored.")
            # window length we need for forecasts is just the
            # length of seasonal periodicity
            self.window_length_ = check_window_length(self.window_length, n_timepoints)
            if self.window_length is None:
                self.window_length_ = len(y)
            if self.window_length == 1:
                raise ValueError(
                    f"For the `drift` strategy, "
                    f"the `window_length`: {self.window_length} "
                    f"value must be greater than one."
                )

        else:
            allowed_strategies = ("last", "mean", "drift")
            raise ValueError(
                f"Unknown strategy: {self.strategy}. Expected "
                f"one of: {allowed_strategies}."
            )

        # check window length
        if self.window_length_ > len(self._y):
            param = (
                "sp" if self.strategy == "last" and self.sp != 1 else "window_length_"
            )
            raise ValueError(
                f"The {param}: {self.window_length_} is larger than "
                f"the training series."
            )

        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Internal predict"""
        last_window, _ = self._get_last_window()
        fh = fh.to_relative(self.cutoff)

        # if last window only contains missing values, return nan
        if np.all(np.isnan(last_window)) or len(last_window) == 0:
            return self._predict_nan(fh)

        elif self.strategy == "last":
            if self.sp == 1:
                return np.repeat(last_window[-1], len(fh))

            else:
                # we need to replicate the last window if max(fh) is larger
                # than sp,so that we still make forecasts by repeating the
                # last value for that season, assume fh is sorted, i.e. max(
                # fh) == fh[-1]
                if fh[-1] > self.sp_:
                    reps = np.int(np.ceil(fh[-1] / self.sp_))
                    last_window = np.tile(last_window, reps=reps)

                # get zero-based index by subtracting the minimum
                fh_idx = fh.to_indexer(self.cutoff)
                return last_window[fh_idx]

        elif self.strategy == "mean":
            if self.sp == 1:
                return np.repeat(np.nanmean(last_window), len(fh))

            else:
                # if the window length is not a multiple of sp, we pad the
                # window with nan values for easy computation of the mean
                remainder = self.window_length_ % self.sp_
                if remainder > 0:
                    pad_width = self.sp_ - remainder
                else:
                    pad_width = 0
                last_window = np.hstack([np.full(pad_width, np.nan), last_window])

                # reshape last window, one column per season
                last_window = last_window.reshape(
                    np.int(np.ceil(self.window_length_ / self.sp_)), self.sp_
                )

                # compute seasonal mean, averaging over rows
                y_pred = np.nanmean(last_window, axis=0)

                # we need to replicate the last window if max(fh) is
                # larger than sp,
                # so that we still make forecasts by repeating the
                # last value for that season,
                # assume fh is sorted, i.e. max(fh) == fh[-1]
                # only slicing all the last seasons into last_window
                if fh[-1] > self.sp_:
                    reps = np.int(np.ceil(fh[-1] / self.sp_))
                    y_pred = np.tile(y_pred, reps=reps)

                # get zero-based index by subtracting the minimum
                fh_idx = fh.to_indexer(self.cutoff)
                return y_pred[fh_idx]

        # if self.strategy == "drift":
        else:
            if self.window_length_ != 1:
                if np.any(np.isnan(last_window[[0, -1]])):
                    raise ValueError(
                        f"For {self.strategy},"
                        f"first and last elements in the last "
                        f"window must not be a missing value."
                    )
                else:
                    # formula for slope
                    slope = (last_window[-1] - last_window[0]) / (
                        self.window_length_ - 1
                    )

                    # get zero-based index by subtracting the minimum
                    fh_idx = fh.to_indexer(self.cutoff)

                    # linear extrapolation
                    y_pred = last_window[-1] + (fh_idx + 1) * slope
                    return y_pred
