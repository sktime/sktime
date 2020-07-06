#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = ["NaiveForecaster"]
__author__ = "Markus Löning"

from warnings import warn

import numpy as np
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

        * "last" and sp is None: forecast the last value in the training series
        * "mean" and sp is None: forecast the mean of (a given window) of the training series
        * "last" and sp is not None: forecasts the last value of the same season in the training series
        * "mean" and sp is not None: forecast the mean (of a given window) of the same season in the training series
    sp : int or None, optional (default=None)
        Seasonal periodicity to use in the seasonal forecast strategies. If None, naive strategy will be used
=======
class NaiveForecaster(OptionalForecastingHorizonMixin,
                      BaseLastWindowForecaster):
    """
    NaiveForecaster is a forecaster that makes forecasts using simple
    strategies.

    Parameters
    ----------
    strategy : str{"last", "mean", "seasonal_last"}, optional (default="last")
        Strategy used to make forecasts:

        * "last": forecast the last value in the training series
        * "mean": forecast the mean of (a given window) of the training series
        * "seasonal_last": forecasts the last value of the same season in
        the training series

>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
    window_length : int or None, optional (default=None)
        Window length to use in the `mean` strategy. If None, entire training
            series will be used.
    """

<<<<<<< HEAD
    def __init__(self, strategy = "last", window_length = None, sp = None):
        super(NaiveForecaster, self).__init__()
        # input checks
        # allowed strategies to include: last, constant, mean, median, seasonal-last, seasonal-mean
        allowed_strategies = ("last", "mean")
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy: {strategy}; expected one of {allowed_strategies}")
=======
    def __init__(self, strategy="last", window_length=None, sp=None):
        super(NaiveForecaster, self).__init__()
        # input checks
        # allowed strategies to include: last, constant, seasonal-last,
        # mean, median
        allowed_strategies = ("last", "mean", "seasonal_last")
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

        if self.strategy == "last" :
            if self.window_length is not None:
                warn("For the `last` strategy, "
                     "the `window_length_` value will be ignored.")

        if self.strategy == "last" and self.sp is None:
            self.window_length_ = 1

        if self.strategy == "last" and self.sp is not None:
            self.sp_ = check_sp(self.sp)
=======
        """  # X_train is ignored
        self._set_oh(y_train)
        self._set_fh(fh)

        if self.strategy in ("last", "seasonal_last"):
            if self.window_length is not None:
                warn("For the `last` and `seasonal_last` strategy, "
                     "the `window_length_` value will be ignored.")

        if self.strategy in ("last", "mean"):
            if self.sp is not None:
                warn("For the `last` and `mean` strategy, "
                     "the `sp` value will be ignored.")

        if self.strategy == "last":
            self.window_length_ = 1

        if self.strategy == "seasonal_last":
            if self.sp is None:
                raise NotImplementedError(
                    "Automatic estimation of the seasonal periodicity `sp` "
                    "from the data is not implemented yet; "
                    "please specify the `sp` value.")
            self.sp_ = check_sp(self.sp)

>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
            # window length we need for forecasts is just the
            # length of seasonal periodicity
            self.window_length_ = self.sp_

        if self.strategy == "mean":
<<<<<<< HEAD
            # check window length is greater than sp for seasonal mean
            if self.window_length is not None and self.sp is not None:
                if self.window_length < self.sp:
                    param1, param2 = "window_length", "sp"
                    raise ValueError(f"The {param1}: {self.window_length} is lesser than "
                                     f"the {param2}: {self.sp}.")

=======
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
            self.window_length_ = check_window_length(self.window_length)

        #  if not given, set default window length for the mean strategy
        if self.strategy == "mean" and self.window_length is None:
            self.window_length_ = len(y_train)

        # check window length
        if self.window_length_ > len(self.oh):
            param = "sp" if self.strategy == "seasonal_last" else \
                "window_length_"
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

<<<<<<< HEAD
        elif self.strategy == "last" and self.sp is None:
            return np.repeat(last_window[-1], len(fh))

        elif self.strategy == "seasonal_last":
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

<<<<<<< HEAD
        elif self.strategy == "mean" and self.sp is None:
            return np.repeat(np.nanmean(last_window), len(fh))

        elif self.strategy == "mean" and self.sp is not None:
            last_window = pd.DataFrame(data = last_window, columns=['data'])
            for i in range(self.sp_):
                if any(last_window.index % self.sp_ == i) is True:
                    last_window.at[last_window[last_window.index % self.sp_ == i].index[-1], 'data'] =\
                        last_window[last_window.index % self.sp_ == i].mean()
            
            # we need to replicate the last window if max(fh) is larger than sp,
            # so that we still make forecasts by repeating the last value for that season,
            # assume fh is sorted, i.e. max(fh) == fh[-1]
            if fh[-1] > self.sp_:
                reps = np.int(np.ceil(fh[-1] / self.sp_))
                last_window = np.tile(last_window['data'].tail(self.sp_).to_numpy(), reps=reps)
            else:
                last_window=last_window.tail(self.sp_).tail(self.sp_).to_numpy()

            # get zero-based index by subtracting the minimum
            fh_idx = fh.index_like(self.cutoff)
            return last_window[fh_idx]    


