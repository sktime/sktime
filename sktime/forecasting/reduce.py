#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"
__all__ = [
    "ReducedTimeSeriesRegressionForecaster",
    "DirectTimeSeriesRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "ReducedRegressionForecaster",
    "DirectRegressionForecaster",
    "RecursiveRegressionForecaster"
]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import clone
from sktime.forecasting.base import DEFAULT_ALPHA
from sktime.forecasting.base import _BaseForecaster
from sktime.forecasting.base import _BaseForecasterOptionalFHinFit
from sktime.forecasting.base import _BaseForecasterRequiredFHinFit
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.utils.validation.forecasting import check_cv
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_time_index
from sktime.utils.validation.forecasting import check_y
from sktime.utils.validation.forecasting import check_window_length


##############################################################################
# base classes for reduction from forecasting to regression

class _BaseReducer(_BaseForecaster):
    """Base class for reducing forecasting to time series regression"""

    _required_parameters = ["regressor", "cv"]

    def __init__(self, regressor, cv):
        self.regressor = regressor
        self.cv = check_cv(cv)

        # no need to validate window length, has been
        # validated already in CV object
        self._window_length = cv.window_length

        self._last_window = None

        super(_BaseReducer, self).__init__()

    def transform(self, y_train):
        """Transform data using rolling window approach"""

        # get integer time index
        time_index = check_time_index(y_train)
        cv = self.cv

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

    def update(self, y_new, X_new=None, update_params=False):
        if X_new is not None:
            raise NotImplementedError()
        if update_params:
            raise NotImplementedError()

        # input checks
        self._check_is_fitted()

        y_new = check_y(y_new)

        # update observation horizon
        self._set_oh(y_new)
        return self

    def predict_in_sample(self, y_train, fh=None, X_train=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Make in-sample predictions"""

        # input checks
        self._check_is_fitted()
        if fh is not None:
            fh = check_fh(fh)

        # get parameters
        window_length = self._window_length
        n_timepoints = len(self.oh)

        # initialise array for predictions
        y_pred = np.zeros(n_timepoints)
        y_pred[:window_length] = np.nan

        # initialise last window
        self._last_window = y_train[:window_length]

        # iterate over training series
        cv = SlidingWindowSplitter(fh=1, window_length=window_length)
        for k, (i, o) in enumerate(cv.split(y_train.index), start=window_length):
            y_new = y_train.iloc[i]
            self._last_window = np.append(self._last_window, y_new)[-self._window_length:]
            y_pred[k] = self.predict(fh=1, return_pred_int=return_pred_int, alpha=alpha)

        # select only predictions in given fh
        fh_idx = fh - np.min(fh)
        return pd.Series(y_pred, index=y_train.index).iloc[fh_idx]


class _ReducedTimeSeriesRegressorMixin:
    """Base class for reducing forecasting to time series regression"""

    @staticmethod
    def _convert_data(X, y=None):
        """Helper function to combine windows from temporal cross-validation into nested
        pandas DataFrame used for solving forecasting via reduction to time series regression.

        Parameters
        ----------
        X : list of pd.Series or np.ndarray
        y : list of pd.Series or np.ndarray, optional (default=None)

        Returns
        -------
        X_train : pd.DataFrame
            nested time series DataFrame
        y_train : np.ndarray
        """
        # return nested dataframe
        X_train = pd.DataFrame(pd.Series([np.asarray(xi) for xi in X]))
        if y is None:
            return X_train

        y_train = np.array([np.asarray(yi) for yi in y])
        return X_train, y_train


class _ReducedTabularRegressorMixin:
    """Base class for reducing forecasting to tabular regression"""

    @staticmethod
    def _convert_data(X, y=None):
        """Helper function to combine windows from temporal cross-validation into numpy array
        used for solving forecasting via reduction to tabular regression.

        Parameters
        ----------
        X : list of pd.Series or np.ndarray
        y : list of pd.Series or np.ndarray, optional (default=None)

        Returns
        -------
        X_train : np.ndarray
        y_train : np.ndarray
        """
        X_train = np.vstack(X)
        if y is None:
            return X_train

        y_train = np.vstack(y)
        return X_train, y_train


class _DirectReducer(_BaseReducer, _BaseForecasterRequiredFHinFit):
    strategy = "direct"

    def __init__(self, regressor, cv):
        super(_DirectReducer, self).__init__(regressor=regressor, cv=cv)

    def fit(self, y_train, fh=None, X_train=None):
        # input checks
        if X_train is not None:
            raise NotImplementedError()

        y_train = check_y(y_train)
        self._set_oh(y_train)
        self._last_window = y_train.values[-self._window_length:]
        self._set_fh(fh)

        # for the direct reduction strategy, a separate forecaster is fitted
        # for step ahead of the forecasting horizon; raise warning if fh in cv
        # does not contain fh passed to fit
        if not np.array_equal(self.cv.fh, self.fh):
            warn(f"The `fh` of the temporal cross validator `cv` must contain the "
                 f"`fh` passed to `fit`, the `fh` of the `cv` will be ignored and "
                 f"the `fh` passed to `fit` will be used instead.")
            self.cv._fh = self.fh

        # transform data using rolling window split
        X_train, Y_train = self.transform(y_train)

        # iterate over forecasting horizon
        self.regressors_ = []
        for i in range(len(self.fh)):
            y_train = Y_train[:, i]
            regressor = clone(self.regressor)
            regressor.fit(X_train, y_train)
            self.regressors_.append(regressor)

        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):

        # check input
        if X is not None:
            raise NotImplementedError()
        if return_pred_int:
            raise NotImplementedError()

        self._check_is_fitted()
        self._set_fh(fh)

        # use last window as new input data for time series regressors to make forecasts
        X_last = self._convert_data([self._last_window])

        # preallocate array for forecasted values
        len_fh = len(self.fh)
        y_pred = np.zeros(len_fh)

        # Iterate over estimators/forecast horizon
        for i, regressor in enumerate(self.regressors_):
            y_pred[i] = regressor.predict(X_last)

        # Add forecasting time index
        index = self._get_absolute_fh()
        return pd.Series(y_pred, index=index)


class _RecursiveReducer(_BaseReducer, _BaseForecasterOptionalFHinFit):
    strategy = "recursive"

    def __init__(self, regressor, cv):

        # for recursive reduction strategy, the forecaster are fitted on
        # one-step ahead forecast and then recursively applied to make
        # forecasts for the whole forecasting horizon, raise warning if
        # different fh is given in cv
        if not np.array_equal(cv.fh, np.array([1])):
            warn(f"The `fh` of the temporal cross validator `cv` must be 1, but found: {cv.fh}. "
                 f"This value will be ignored and 1 will be used instead.")
            cv.fh = np.array([1])

        self.regressor_ = None
        super(_RecursiveReducer, self).__init__(regressor=regressor, cv=cv)

    def fit(self, y_train, fh=None, X_train=None):
        """Fit forecaster"""
        # input checks
        if X_train is not None:
            raise NotImplementedError()

        y_train = check_y(y_train)

        # set values
        self._set_fh(fh)
        self._set_oh(y_train)
        self._last_window = y_train.values[-self._window_length:]

        # transform data
        X_train, y_train = self.transform(y_train)

        # fit base regressor
        regressor = clone(self.regressor)
        regressor.fit(X_train, y_train)
        self.regressor_ = regressor

        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Predict"""
        # check inputs
        if X is not None:
            raise NotImplementedError()
        if return_pred_int:
            raise NotImplementedError()

        self._check_is_fitted()
        self._set_fh(fh)

        # prepare recursive predictions
        fh_max = max(self.fh)
        y_pred = np.zeros(fh_max)

        regressor = self.regressor_

        # recursively predict iterating over forecasting horizon
        for i in range(fh_max):
            X_last = self._convert_data([self._last_window])  # convert data into required input format
            y_pred[i] = regressor.predict(X_last)  # make forecast using fitted regressor
            self._update_last_window(y_pred[i])  # update last window with previous prediction

        # select specific steps ahead and add index
        fh_idx = self.fh - np.min(self.fh)
        index = self._get_absolute_fh()
        return pd.Series(y_pred[fh_idx], index=index)

    def _update_last_window(self, y):
        """Update last window used in recursive predictions"""
        self._last_window = np.append(self._last_window, y)[-self._window_length:]


##############################################################################
# redution to regression
class DirectRegressionForecaster(_DirectReducer, _ReducedTabularRegressorMixin):
    pass


class RecursiveRegressionForecaster(_RecursiveReducer, _ReducedTabularRegressorMixin):
    pass


##############################################################################
# reduction to time series regression
class DirectTimeSeriesRegressionForecaster(_DirectReducer, _ReducedTimeSeriesRegressorMixin):
    pass


class RecursiveTimeSeriesRegressionForecaster(_RecursiveReducer, _ReducedTimeSeriesRegressorMixin):
    pass


##############################################################################
# factory methods for easier user interface, but not tunable as it's not an estimator
def ReducedTimeSeriesRegressionForecaster(ts_regressor, cv, strategy="recursive"):
    """
    Forecasting based on reduction to time series regression.

    When fitting, a rolling window approach is used to first transform the target series into panel data which is
    then used to train a time series regressor. During prediction, the last available data is used as input to the
    fitted time series regressors to make forecasts.

    Parameters
    ----------
    ts_regressor : a time series regressor
    cv : temporal cross-validator
    strategy : str{"direct", "recursive", "dirrec"}, optional (default="direct")

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    scitype = "ts_regressor"
    Forecaster = _get_forecaster_class(scitype, strategy)
    return Forecaster(regressor=ts_regressor, cv=cv)


def ReducedRegressionForecaster(regressor, cv, strategy="recursive"):
    """
    Forecasting based on reduction to tabular regression.

    When fitting, a rolling window approach is used to first transform the target series into panel data which is
    then used to train a regressor. During prediction, the last available data is used as input to the
    fitted regressors to make forecasts.

    Parameters
    ----------
    regressor : a regressor
    cv : temporal cross-validator
    strategy : str{"direct", "recursive", "dirrec"}, optional (default="direct")

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    scitype = "regressor"
    Forecaster = _get_forecaster_class(scitype, strategy)
    return Forecaster(regressor=regressor, cv=cv)


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
