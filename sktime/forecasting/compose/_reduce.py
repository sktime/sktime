#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = "Markus Löning"
__all__ = [
    "ReducedTabularRegressorMixin",
    "ReducedTimeSeriesRegressorMixin",
    "DirectTimeSeriesRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "DirectRegressionForecaster",
    "MultioutputRegressionForecaster",
    "RecursiveRegressionForecaster",
    "ReducedForecaster",
]

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.forecasting.base._sktime import _RequiredForecastingHorizonMixin
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.utils.validation import check_window_length
from sktime.utils.validation.forecasting import check_step_length
from sktime.utils.validation.forecasting import check_y
from sktime.utils.validation.forecasting import check_X


##############################################################################
# Lovkush's new classes
# rather than creating various base classes and mixins, i am going to create
# each final class using only _BaseWindowForecaster and ForecastingHorizonMixins
# The intended benefits:
# - help me better understand exactly what goes into each reductionforecaster by
#   making  it explicit
# - reveal explicitly what different reducers do and do not have in common.
# - make it easier to change one forecaster without having to worry about whether
#   it will break other forecasting. e.g. if one wanted to add in-sample predictions,
#   can get it to work on one forecaster first, rather than having to make it work
#   on all of them at once.
# - make it more user-friendly for future contributors.
# - ensure new classes are consistent with old ones
# Possible downside:
# - goes against design principle of avoiding copying and pasting.
#   however, current system requires a lot of copying and pasting already,
#   so this is only downside against some hypothetical perfect code


class NewDirectRegressionForecaster(
    _BaseWindowForecaster, _RequiredForecastingHorizonMixin
):
    """
    Forecasting based on reduction to tabular regression with a direct
    reduction strategy.
    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon

    Parameters
    ----------
    regressor : sklearn estimator object
        Define the regression model type.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    _required_parameters = ["regressor"]
    strategy = "direct"

    def __init__(self, regressor, window_length=10, step_length=1):
        super(NewDirectRegressionForecaster, self).__init__(window_length=window_length)
        self.regressor = regressor
        self.step_length = step_length

    def _transform(self, y, X=None, fit_or_predict=None):
        """Transform data using rolling window approach
        Parameters
        ----------
        fit_or_predict : string 'fit' or 'predict'
            Determines whether transform is taking place fit or predict step
        """
        # check inputs and combine into single numpy array
        y = check_y(y)
        y_ = np.asarray(y).reshape(-1, 1)

        if X is None:
            yX_ = y_
        else:
            X = check_X(X)
            yX_ = np.asarray(pd.concat([y, X], axis=1))

        # define variables to help with transformation
        fh_length = len(self.fh)
        w_length = self.window_length
        n_timepoints, _ = yX_.shape

        if self.step_length > 1:
            raise NotImplementedError(
                "Only step-length equal to 1 has been implemented"
            )

        # Transform target series and exogenous variables using rolling window
        if fit_or_predict == "fit":
            reduction_X = np.hstack(
                [
                    yX_[i : i + n_timepoints + 1 - fh_length - w_length]
                    for i in range(w_length)
                ]
            )

            reduction_Y = np.hstack(
                [
                    y_[i + w_length : i + n_timepoints + 1 - fh_length]
                    for i in range(fh_length)
                ]
            )

            return reduction_X, reduction_Y

        elif fit_or_predict == "predict":
            reduction_X = np.hstack(
                [yX_[n_timepoints - w_length + i] for i in range(w_length)]
            )

            return reduction_X

    def fit(self, y, X=None, fh=None):
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
        self._set_y_X(y, X)
        self._set_fh(fh)
        if len(self.fh.to_in_sample(self.cutoff)) > 0:
            raise NotImplementedError("In-sample predictions are not implemented")

        self.step_length = check_step_length(self.step_length)
        self.window_length = check_window_length(self.window_length)

        # transform data
        reduction_X, reduction_Y = self._transform(
            self._y, self._X, fit_or_predict="fit"
        )

        # iterate over forecasting horizon
        self.regressors_ = []
        for i in range(len(self.fh)):
            reduction_y = reduction_Y[:, i]
            regressor = clone(self.regressor)
            regressor.fit(reduction_X, reduction_y)
            self.regressors_.append(regressor)

        self._is_fitted = True
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # get last window from self._transform
        reduction_X_last = self._transform(
            self._y, self._X, fit_or_predict="prediction"
        )

        # preallocate array for forecasted values
        y_pred = np.zeros(len(fh))

        # Iterate over estimators/forecast horizon
        for i, regressor in enumerate(self.regressors_):
            y_pred[i] = regressor.predict(reduction_X_last)
        return y_pred

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        # it's not clear how the direct reducer would generate in-sample
        # predictions
        raise NotImplementedError("in-sample predictions are not implemented")


##############################################################################
# base classes for reduction from forecasting to regression


class BaseReducer(_BaseWindowForecaster):
    """Base class for reducing forecasting to time series regression"""

    _required_parameters = ["regressor"]

    def __init__(self, regressor, window_length=10, step_length=1):
        super(BaseReducer, self).__init__(window_length=window_length)
        self.regressor = regressor
        self.step_length = step_length
        self.step_length_ = None
        self._cv = None

    def _transform(self, y, X=None):
        """Transform data using rolling window approach"""
        if X is not None:
            raise NotImplementedError("Exogenous variables `X` are not yet supported.")
        y = check_y(y)

        # get integer time index
        cv = self._cv

        # Transform target series into tabular format using
        # rolling window tabularisation
        x_windows = []
        y_windows = []
        for x_index, y_index in cv.split(y):
            x_window = y.iloc[x_index]
            y_window = y.iloc[y_index]

            x_windows.append(x_window)
            y_windows.append(y_window)

        # Put into required input format for regression
        X, y = self._format_windows(x_windows, y_windows)
        return X, y

    def _format_windows(self, x_windows, y_windows=None):
        """Helper function to combine windows from temporal cross-validation
        into nested
        pd.DataFrame for reduction to time series regression or tabular
        np.array for
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
        X = self._format_x_windows(x_windows)

        # during prediction, y=None, so only return X
        if y_windows is None:
            return X

        y = self._format_y_windows(y_windows)
        return X, y

    @staticmethod
    def _format_y_windows(y_windows):
        """Template method for formatting y windows"""
        raise NotImplementedError("abstract method")

    @staticmethod
    def _format_x_windows(x_windows):
        """Template method for formatting x windows"""
        raise NotImplementedError("abstract method")

    def _is_predictable(self, last_window):
        """Helper function to check if we can make predictions from last
        window"""
        return (
            len(last_window) == self.window_length_
            and np.sum(np.isnan(last_window)) == 0
            and np.sum(np.isinf(last_window)) == 0
        )


class ReducedTimeSeriesRegressorMixin:
    """Mixin class for reducing forecasting to time series regression"""

    @staticmethod
    def _format_x_windows(x_windows):
        """Helper function to combine windows from temporal cross-validation
        into nested
        pd.DataFrame used for solving forecasting via reduction to time
        series regression.

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
    def _format_y_windows(y_windows):
        return np.array([np.asarray(yi) for yi in y_windows])


class ReducedTabularRegressorMixin:
    """Mixin class for reducing forecasting to tabular regression"""

    @staticmethod
    def _format_x_windows(x_windows):
        """Helper function to combine windows from temporal cross-validation
        into nested
        pd.DataFrame used for solving forecasting via reduction to time
        series regression.

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
    def _format_y_windows(y_windows):
        return np.vstack(y_windows)


class _DirectReducer(_RequiredForecastingHorizonMixin, BaseReducer):
    strategy = "direct"

    def fit(self, y, X=None, fh=None):
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
        self._set_y_X(y, X)
        if X is not None:
            raise NotImplementedError("Exogenous variables `X` are not yet supported.")
        self._set_fh(fh)
        if len(self.fh.to_in_sample(self.cutoff)) > 0:
            raise NotImplementedError("In-sample predictions are not implemented")

        self.step_length_ = check_step_length(self.step_length)
        self.window_length_ = check_window_length(self.window_length)

        # for the direct reduction strategy, a separate forecaster is fitted
        # for each step ahead of the forecasting horizon
        self._cv = SlidingWindowSplitter(
            fh=self.fh.to_relative(self.cutoff),
            window_length=self.window_length_,
            step_length=self.step_length_,
            start_with_window=True,
        )

        # transform data using rolling window split
        X, Y_train = self._transform(y, X)

        # iterate over forecasting horizon
        self.regressors_ = []
        for i in range(len(self.fh)):
            y = Y_train[:, i]
            regressor = clone(self.regressor)
            regressor.fit(X, y)
            self.regressors_.append(regressor)

        self._is_fitted = True
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # use last window as new input data for time series regressors to
        # make forecasts
        # get last window from observation horizon
        last_window, _ = self._get_last_window()
        if not self._is_predictable(last_window):
            return self._predict_nan(fh)

        X_last = self._format_windows([last_window])

        # preallocate array for forecasted values
        y_pred = np.zeros(len(fh))

        # Iterate over estimators/forecast horizon
        for i, regressor in enumerate(self.regressors_):
            y_pred[i] = regressor.predict(X_last)
        return y_pred

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        # it's not clear how the direct reducer would generate in-sample
        # predictions
        raise NotImplementedError("in-sample predictions are not implemented")


class _MultioutputReducer(_RequiredForecastingHorizonMixin, BaseReducer):
    strategy = "multioutput"

    def fit(self, y, X=None, fh=None):
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
        self._set_y_X(y, X)
        if X is not None:
            raise NotImplementedError("Exogenous variables `X` are not yet supported.")
        self._set_fh(fh)
        if len(self.fh.to_in_sample(self.cutoff)) > 0:
            raise NotImplementedError("In-sample predictions are not implemented")

        self.step_length_ = check_step_length(self.step_length)
        self.window_length_ = check_window_length(self.window_length)

        # for the multioutput reduction strategy, a single forecaster is fitted
        # simultaneously to all the future steps in the forecasting horizon
        # by reducing to a forecaster that can handle multi-dimensional outputs
        self._cv = SlidingWindowSplitter(
            fh=self.fh.to_relative(self.cutoff),
            window_length=self.window_length_,
            step_length=self.step_length_,
            start_with_window=True,
        )

        # transform data using rolling window split
        X, Y_train = self._transform(y, X)

        # fit regressor to training data
        regressor = clone(self.regressor)
        regressor.fit(X, Y_train)
        self.regressor_ = regressor

        self._is_fitted = True
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # use last window as new input data for regressor to
        # make forecasts
        # get last window from observation horizon
        last_window, _ = self._get_last_window()
        if not self._is_predictable(last_window):
            return self._predict_nan(fh)

        X_last = self._format_windows([last_window])

        y_pred = self.regressor_.predict(X_last)

        # preallocate array for forecasted values
        # y_pred = np.zeros(len(fh))

        return y_pred[0]

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        raise NotImplementedError("in-sample predictions are not implemented")


class _RecursiveReducer(_OptionalForecastingHorizonMixin, BaseReducer):
    strategy = "recursive"

    def fit(self, y, X=None, fh=None):
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
        # input checks
        if X is not None:
            raise NotImplementedError("Exogenous variables `X` are not yet supported.")

        # set values
        self._set_y_X(y, X)
        self._set_fh(fh)

        self.step_length_ = check_step_length(self.step_length)
        self.window_length_ = check_window_length(self.window_length)

        # set up cv iterator, for recursive strategy, a single estimator
        # is fit for a one-step-ahead forecasting horizon and then called
        # iteratively to predict multiple steps ahead
        self._cv = SlidingWindowSplitter(
            fh=1,
            window_length=self.window_length_,
            step_length=self.step_length_,
            start_with_window=True,
        )

        # transform data into tabular form
        X_train_tab, y_train_tab = self._transform(y, X)

        # fit base regressor
        regressor = clone(self.regressor)
        regressor.fit(X_train_tab, y_train_tab.ravel())
        self.regressor_ = regressor

        self._is_fitted = True
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Predict"""
        # compute prediction
        # prepare recursive predictions
        fh_max = fh.to_relative(self.cutoff)[-1]
        y_pred = np.zeros(fh_max)

        # get last window from observation horizon
        last_window, _ = self._get_last_window()
        if not self._is_predictable(last_window):
            return self._predict_nan(fh)

        # recursively predict iterating over forecasting horizon
        for i in range(fh_max):
            X_last = self._format_windows(
                [last_window]
            )  # convert data into required input format
            y_pred[i] = self.regressor_.predict(
                X_last
            )  # make forecast using fitted regressor

            # update last window with previous prediction
            last_window = np.append(last_window, y_pred[i])[-self.window_length_ :]

        fh_idx = fh.to_indexer(self.cutoff)
        return y_pred[fh_idx]


##############################################################################
# reduction to regression
class DirectRegressionForecaster(ReducedTabularRegressorMixin, _DirectReducer):
    """
    Forecasting based on reduction to tabular regression with a direct
    reduction strategy.
    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon

    Parameters
    ----------
    regressor : sklearn estimator object
        Define the regression model type.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    pass


class MultioutputRegressionForecaster(
    ReducedTabularRegressorMixin, _MultioutputReducer
):
    """
    Forecasting based on reduction to tabular regression with a multioutput
    reduction strategy.
    For the multioutput reduction strategy, a single forecaster is fitted
    simultaneously to all the future steps in the forecasting horizon

    Parameters
    ----------
    regressor : sklearn estimator object
        Define the regression model type.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    pass


class RecursiveRegressionForecaster(ReducedTabularRegressorMixin, _RecursiveReducer):
    """
    Forecasting based on reduction to tabular regression with a recursive
    reduction strategy.
    For the recursive reduction strategy, a single estimator is
    fit for a one-step-ahead forecasting horizon and then called
    iteratively to predict multiple steps ahead.

    Parameters
    ----------
    regressor : sklearn estimator object
        Define the regression model type.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    pass


##############################################################################
# reduction to time series regression
class DirectTimeSeriesRegressionForecaster(
    ReducedTimeSeriesRegressorMixin, _DirectReducer
):
    """
    Forecasting based on reduction to time series regression with a direct
    reduction strategy.
    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon

    Parameters
    ----------
    regressor : sktime estimator object
        Define the type of time series regression model.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    pass


class RecursiveTimeSeriesRegressionForecaster(
    ReducedTimeSeriesRegressorMixin, _RecursiveReducer
):
    """
    Forecasting based on reduction to time series regression with a recursive
    reduction strategy.
    For the recursive reduction strategy, a single estimator is
    fit for a one-step-ahead forecasting horizon and then called
    iteratively to predict multiple steps ahead.

    Parameters
    ----------
    regressor : sktime estimator object
        Define the type of time series regression model.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    pass


##############################################################################
# factory methods for easier user interface, but not tunable as it's not an
# estimator
def ReducedForecaster(
    regressor, scitype, strategy="recursive", window_length=10, step_length=1
):
    """
    Forecasting based on reduction

    When fitting, a rolling window approach is used to first transform the
    target series into panel data which is
    then used to train a regressor. During prediction, the last
    available data is used as input to the
    fitted regressors to make forecasts.

    Parameters
    ----------
    scitype : str
        Can be 'regressor' or 'ts-regressor'
    strategy : str {"direct", "recursive", "multioutput"}, optional
        Strategy to generate predictions
    window_length : int, optional (default=10)
    step_length : int, optional (default=1)
    regressor : a regressor of type given by parameter scitype

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (
    2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    Forecaster = _get_forecaster_class(scitype, strategy)
    return Forecaster(
        regressor=regressor, window_length=window_length, step_length=step_length
    )


def _get_forecaster_class(scitype, strategy):
    """Helper function to select forecaster for a given scientific type (
    scitype)
    and reduction strategy"""

    allowed_strategies = ("direct", "recursive", "multioutput")
    if strategy not in allowed_strategies:
        raise ValueError(
            f"Unknown strategy, please provide one of {allowed_strategies}."
        )

    if scitype == "ts_regressor" and strategy == "multioutput":
        raise NotImplementedError(
            "The `multioutput` strategy is not yet implemented "
            "for time series regresors."
        )

    lookup_table = {
        "regressor": {
            "direct": DirectRegressionForecaster,
            "recursive": RecursiveRegressionForecaster,
            "multioutput": MultioutputRegressionForecaster,
        },
        "ts_regressor": {
            "direct": DirectTimeSeriesRegressionForecaster,
            "recursive": RecursiveTimeSeriesRegressionForecaster,
        },
    }
    # look up and return forecaster class
    Forecaster = lookup_table.get(scitype).get(strategy)
    return Forecaster
