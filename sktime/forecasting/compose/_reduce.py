#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"
__all__ = [
    "ReducedTabularRegressorMixin",
    "ReducedTimeSeriesRegressorMixin",
    "ReducedTimeSeriesRegressionForecaster",
    "DirectTimeSeriesRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "ReducedRegressionForecaster",
    "DirectRegressionForecaster",
    "RecursiveRegressionForecaster"
]

import numpy as np
import pandas as pd
from sklearn.base import clone
from sktime.forecasting.base import BaseLastWindowForecaster, OptionalForecastingHorizonMixin, \
    RequiredForecastingHorizonMixin
from sktime.forecasting.base import DEFAULT_ALPHA
from sktime.forecasting.model_selection import SlidingWindowSplitter


##############################################################################
# base classes for reduction from forecasting to regression

class BaseReducer(BaseLastWindowForecaster):
    """Base class for reducing forecasting to time series regression"""

    _required_parameters = ["regressor"]

    def __init__(self, regressor, window_length=10, step_length=1):
        super(BaseReducer, self).__init__()
        self.regressor = regressor
        self._window_length = window_length
        self._step_length = step_length
        self._cv = None

    def update(self, y_new, X_new=None, update_params=False):
        if X_new is not None or update_params:
            raise NotImplementedError()

        # input checks
        self._check_is_fitted()

        # update observation horizon
        self._set_oh(y_new)
        return self

    def transform(self, y_train, X_train=None):
        """Transform data using rolling window approach"""

        if X_train is not None:
            raise NotImplementedError()

        # get integer time index
        time_index = y_train.index
        cv = self._cv

        # Transform target series into tabular format using
        # rolling window tabularisation
        x_windows = []
        y_windows = []
        for x_index, y_index in cv.split(time_index):
            x_window = y_train.iloc[x_index]
            x_windows.append(x_window)

            y_window = y_train.iloc[y_index]
            y_windows.append(y_window)

        # Put into required input format for regression
        X_train, y_train = self._convert_data(x_windows, y_windows)
        return X_train, y_train

    def _convert_data(self, x_windows, y_windows=None):
        """Helper function to combine windows from temporal cross-validation into nested
        pd.DataFrame for reduction to time series regression or tabular np.array for
        tabular regression.

        Parameters
        ----------
        x_windows : list of pd.Series or np.array
        y_windows : list of pd.Series or np.array, optional (default=None)

        Returns
        -------
        X : pd.DataFrame or np.array
            Nested time series data frame.
        y : np.array
            Array of target values.
        """
        X = self._convert_x_windows(x_windows)

        # during prediction, y=None, so only return X
        if y_windows is None:
            return X

        y = self._convert_y_windows(y_windows)
        return X, y

    @staticmethod
    def _convert_y_windows(y):
        raise NotImplementedError("abstract method")

    @staticmethod
    def _convert_x_windows(X):
        raise NotImplementedError("abstract method")

    @property
    def step_length(self):
        return self._step_length


class ReducedTimeSeriesRegressorMixin:
    """Mixin class for reducing forecasting to time series regression"""

    @staticmethod
    def _convert_x_windows(x_windows):
        """Helper function to combine windows from temporal cross-validation into nested
        pd.DataFrame used for solving forecasting via reduction to time series regression.

        Parameters
        ----------
        x_windows : list of pd.Series or np.array

        Returns
        -------
        X : pd.DataFrame
            Nested time series data frame.
        """
        # return nested dataframe
        return pd.DataFrame(pd.Series([pd.Series(xi) for xi in x_windows]))

    @staticmethod
    def _convert_y_windows(y_windows):
        return np.array([np.asarray(yi) for yi in y_windows])


class ReducedTabularRegressorMixin:
    """Mixin class for reducing forecasting to tabular regression"""

    @staticmethod
    def _convert_x_windows(x_windows):
        """Helper function to combine windows from temporal cross-validation into nested
        pd.DataFrame used for solving forecasting via reduction to time series regression.

        Parameters
        ----------
        x_windows : list of pd.Series or np.array

        Returns
        -------
        X : pd.DataFrame
            Nested time series data frame.
        """
        return np.vstack(x_windows)

    @staticmethod
    def _convert_y_windows(y_windows):
        return np.vstack(y_windows)


class _DirectReducer(RequiredForecastingHorizonMixin, BaseReducer):
    strategy = "direct"

    def fit(self, y_train, fh=None, X_train=None):
        # input checks
        if X_train is not None:
            raise NotImplementedError()

        self._set_oh(y_train)
        self._set_fh(fh)

        # for the direct reduction strategy, a separate forecaster is fitted
        # for each step ahead of the forecasting horizon
        self._cv = SlidingWindowSplitter(fh=self.fh, window_length=self.window_length, step_length=self.step_length,
                                         start_with_window=True)

        # transform data using rolling window split
        X_train, Y_train = self.transform(y_train, X_train)

        # iterate over forecasting horizon
        self.regressors_ = []
        for i in range(len(self.fh)):
            y_train = Y_train[:, i]
            regressor = clone(self.regressor)
            regressor.fit(X_train, y_train)
            self.regressors_.append(regressor)

        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        # use last window as new input data for time series regressors to make forecasts
        # get last window from observation horizon
        last_window = self._get_last_window()
        X_last = self._convert_data([last_window])

        # preallocate array for forecasted values
        y_pred = np.zeros(len(fh))

        # Iterate over estimators/forecast horizon
        for i, regressor in enumerate(self.regressors_):
            y_pred[i] = regressor.predict(X_last)
        return y_pred

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        # it's not clear how the direct reducer would generate in-sample predictions
        raise NotImplementedError("in-sample predictions are not implemented")


class _RecursiveReducer(OptionalForecastingHorizonMixin, BaseReducer):
    strategy = "recursive"

    def fit(self, y_train, fh=None, X_train=None):
        """Fit forecaster"""
        # input checks
        if X_train is not None:
            raise NotImplementedError()

        # set values
        self._set_oh(y_train)
        self._set_fh(fh)

        # set up cv iterator, for recursive strategy, a single estimator
        # is fit for a one-step-ahead forecasting horizon and then called
        # iteratively to predict multiple steps ahead
        self._cv = SlidingWindowSplitter(fh=1, window_length=self.window_length, step_length=self.step_length,
                                         start_with_window=True)

        # transform data
        X_train, y_train = self.transform(y_train, X_train)
        y_train = y_train.ravel()

        # fit base regressor
        regressor = clone(self.regressor)
        regressor.fit(X_train, y_train)
        self.regressor_ = regressor

        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Predict"""
        # compute prediction
        # prepare recursive predictions
        fh_max = fh[-1]
        y_pred = np.zeros(fh_max)
        regressor = self.regressor_

        # get last window from observation horizon
        last_window = self._get_last_window()

        # recursively predict iterating over forecasting horizon
        for i in range(fh_max):
            X_last = self._convert_data([last_window])  # convert data into required input format
            y_pred[i] = regressor.predict(X_last)  # make forecast using fitted regressor

            # update last window with previous prediction
            last_window = np.append(last_window, y_pred[i])[-self._window_length:]

        fh_idx = self._get_array_index_fh(fh)
        return y_pred[fh_idx]

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        raise NotImplementedError("in-sample predictions are not implemented")


##############################################################################
# redution to regression
class DirectRegressionForecaster(ReducedTabularRegressorMixin, _DirectReducer):
    pass


class RecursiveRegressionForecaster(ReducedTabularRegressorMixin, _RecursiveReducer):
    pass


##############################################################################
# reduction to time series regression
class DirectTimeSeriesRegressionForecaster(ReducedTimeSeriesRegressorMixin, _DirectReducer):
    pass


class RecursiveTimeSeriesRegressionForecaster(ReducedTimeSeriesRegressorMixin, _RecursiveReducer):
    pass


##############################################################################
# factory methods for easier user interface, but not tunable as it's not an estimator
def ReducedTimeSeriesRegressionForecaster(ts_regressor, strategy="recursive", window_length=10, step_length=1):
    """
    Forecasting based on reduction to time series regression.

    When fitting, a rolling window approach is used to first transform the target series into panel data which is
    then used to train a time series regressor. During prediction, the last available data is used as input to the
    fitted time series regressors to make forecasts.

    Parameters
    ----------
    ts_regressor : a time series regressor

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    scitype = "ts_regressor"
    Forecaster = _get_forecaster_class(scitype, strategy)
    return Forecaster(regressor=ts_regressor, window_length=window_length, step_length=step_length)


def ReducedRegressionForecaster(regressor, strategy="recursive", window_length=10, step_length=1):
    """
    Forecasting based on reduction to tabular regression.

    When fitting, a rolling window approach is used to first transform the target series into panel data which is
    then used to train a regressor. During prediction, the last available data is used as input to the
    fitted regressors to make forecasts.

    Parameters
    ----------
    regressor : a regressor

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    scitype = "regressor"
    Forecaster = _get_forecaster_class(scitype, strategy)
    return Forecaster(regressor=regressor, window_length=window_length, step_length=step_length)


def _get_forecaster_class(scitype, strategy):
    """Helper function to select forecaster for a given scientific type (scitype)
    and reduction strategy"""

    allowed_strategies = ("direct", "recursive", "dirrec")
    if strategy not in allowed_strategies:
        raise ValueError(f"Unknown strategy, please provide one of {allowed_strategies}.")

    if strategy == "dirrec":
        raise NotImplementedError("The `dirrec` strategy is not yet implemented.")

    lookup_table = {
        "regressor": {
            "direct": DirectRegressionForecaster,
            "recursive": RecursiveRegressionForecaster,
        },
        "ts_regressor": {
            "direct": DirectTimeSeriesRegressionForecaster,
            "recursive": RecursiveTimeSeriesRegressionForecaster
        }
    }
    # look up and return forecaster class
    Forecaster = lookup_table.get(scitype).get(strategy)
    return Forecaster
