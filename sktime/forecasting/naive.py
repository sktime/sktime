# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements simple forecasts based on naive assumptions."""

__all__ = ["NaiveForecaster"]
__author__ = ["mloning", "Piyush Gade", "Flix6x", "aiwalter"]

from warnings import warn

import numpy as np

from sktime.forecasting.base._base import DEFAULT_ALPHA, BaseForecaster
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.utils.validation import check_window_length
from sktime.utils.validation.forecasting import check_sp


class _NaiveForecaster(_BaseWindowForecaster):
    """Univariate NaiveForecaster."""

    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": True,  # todo: switch to True if GH1367 is fixed
        "scitype:y": "univariate",
    }

    def __init__(self, strategy="last", window_length=None, sp=1):
        super(_NaiveForecaster, self).__init__()
        self.strategy = strategy
        self.sp = sp
        self.window_length = window_length

        # Override tag for handling missing data
        # todo: remove if GH1367 is fixed
        if self.strategy in ("last", "mean"):
            self.set_tags(**{"handles-missing-data": True})

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, default=None
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        # X_train is ignored

        n_timepoints = y.shape[0]

        if self.strategy in ("last", "mean"):
            # check window length is greater than sp for seasonal mean or seasonal last
            if self.window_length is not None and self.sp != 1:
                if self.window_length < self.sp:
                    raise ValueError(
                        f"The `window_length`: "
                        f"{self.window_length} is smaller than "
                        f"`sp`: {self.sp}."
                    )
            self.window_length_ = check_window_length(self.window_length, n_timepoints)
            self.sp_ = check_sp(self.sp)

            #  if not given, set default window length
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
        """Calculate predictions for use in predict."""
        last_window, _ = self._get_last_window()
        fh = fh.to_relative(self.cutoff)

        strategy = self.strategy
        sp = self.sp

        # if last window only contains missing values, return nan
        if np.all(np.isnan(last_window)) or len(last_window) == 0:
            return self._predict_nan(fh)

        elif strategy == "last":
            if sp == 1:
                last_valid_value = last_window[
                    (~np.isnan(last_window))[0::sp].cumsum().argmax()
                ]
                return np.repeat(last_valid_value, len(fh))

            else:
                # reshape last window, one column per season
                last_window = self._reshape_last_window_for_sp(last_window)

                # select last non-NaN row for every column
                y_pred = last_window[
                    (~np.isnan(last_window)).cumsum(0).argmax(0).T,
                    range(last_window.shape[1]),
                ]

                # tile prediction according to seasonal periodicity
                y_pred = self._tile_seasonal_prediction(y_pred, fh)

        elif strategy == "mean":
            if sp == 1:
                return np.repeat(np.nanmean(last_window), len(fh))

            else:
                # reshape last window, one column per season
                last_window = self._reshape_last_window_for_sp(last_window)

                # compute seasonal mean, averaging over non-NaN rows for every column
                y_pred = np.nanmean(last_window, axis=0)

                # tile prediction according to seasonal periodicity
                y_pred = self._tile_seasonal_prediction(y_pred, fh)

        elif strategy == "drift":
            if self.window_length_ != 1:
                if np.any(np.isnan(last_window[[0, -1]])):
                    raise ValueError(
                        f"For {strategy},"
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
        else:
            raise ValueError(f"unknown strategy {strategy} provided to NaiveForecaster")

        return y_pred

    def _reshape_last_window_for_sp(self, last_window):
        """Reshape the 1D last window into a 2D last window, prepended with NaN values.

        The 2D array has 1 column per season.

        For example:

            last_window = [1, 2, 3, 4]
            sp = 3  # i.e. 3 distinct seasons
            reshaped_last_window = [[nan, nan, 1],
                                    [  2,   3, 4]]
        """
        # if window length is not multiple of sp, backward fill window with nan values
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

        return last_window

    def _tile_seasonal_prediction(self, y_pred, fh):
        """Tile a prediction to cover all requested forecasting horizons.

        The original prediction has 1 value per season.

        For example:

            fh = [1, 2, 3, 4, 5, 6, 7]
            y_pred = [2, 3, 1]  # note len(y_pred) = sp
            y_pred_tiled = [2, 3, 1, 2, 3, 1, 2]
        """
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


class NaiveForecaster(BaseForecaster):
    """Forecast based on naive assumptions about past trends continuing.

    NaiveForecaster is a forecaster that makes forecasts using simple
    strategies. Two out of three strategies are robust against NaNs. The
    NaiveForecaster can also be used for multivariate data and it then
    applies internally the ColumnEnsembleForecaster, so each column
    is forecasted with the same strategy.

    Internally, this forecaster does the following:
    - obtains the so-called "last window", a 1D array that denotes the
      most recent time window that the forecaster is allowed to use
    - reshapes the last window into a 2D array according to the given
      seasonal periodicity (prepended with NaN values to make it fit);
    - make a prediction for each column, using the given strategy:
      - "last": last non-NaN row
      - "mean": np.nanmean over rows
    - tile the predictions using the seasonal periodicity

    Parameters
    ----------
    strategy : {"last", "mean", "drift"}, default="last"
        Strategy used to make forecasts:

        * "last":   (robust against NaN values)
                    forecast the last value in the
                    training series when sp is 1.
                    When sp is not 1,
                    last value of each season
                    in the last window will be
                    forecasted for each season.
        * "mean":   (robust against NaN values)
                    forecast the mean of last window
                    of training series when sp is 1.
                    When sp is not 1, mean of all values
                    in a season from last window will be
                    forecasted for each season.
        * "drift":  (not robust against NaN values)
                    forecast by fitting a line between the
                    first and last point of the window and
                    extrapolating it into the future.

    sp : int, default=1
        Seasonal periodicity to use in the seasonal forecasting.

    window_length : int or None, default=None
        Window length to use in the `mean` strategy. If None, entire training
            series will be used.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> forecaster.fit(y)
    NaiveForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _tags = {
        "y_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "scitype:y": "both",
        "requires-fh-in-fit": False,
        "handles-missing-data": True,  # todo: switch to True if GH1367 is fixed
    }

    def __init__(self, strategy="last", window_length=None, sp=1):
        self.strategy = strategy
        self.sp = sp
        self.window_length = window_length
        super(NaiveForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, default=None
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, default=None
            Exogenous variables are ignored.

        Returns
        -------
        self : returns an instance of self.
        """
        self._forecaster = ColumnEnsembleForecaster(
            _NaiveForecaster(
                strategy=self.strategy, sp=self.sp, window_length=self.window_length
            )
        )
        self._forecaster.fit(y=y, X=X, fh=fh)

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        """
        y_pred = self._forecaster.predict(
            fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha
        )

        # check for in-sample prediction, if first time point needs to be imputed
        if self._y.index[0] in y_pred.index:
            # fill NaN with next row values
            y_pred.loc[self._y.index[0]] = y_pred.loc[self._y.index[1]]

        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update cutoff value and, optionally, fitted parameters.

        Parameters
        ----------
        y : pd.Series, pd.DataFrame, or np.array
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogeneous data
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        return self._forecaster.update(y=y, X=X, update_params=update_params)
