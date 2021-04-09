#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Lovkush Agarwal", "Markus Löning"]
__all__ = [
    "make_reduction",
    "DirectTimeSeriesRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "MultioutputTimeSeriesRegressionForecaster",
    "DirectTabularRegressionForecaster",
    "RecursiveTabularRegressionForecaster",
    "MultioutputTabularRegressionForecaster",
    "ReducedForecaster",
    "ReducedRegressionForecaster",
]

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.base import clone

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.forecasting.base._sktime import _RequiredForecastingHorizonMixin
from sktime.regression.base import BaseRegressor
from sktime.utils._maint import deprecated
from sktime.utils.validation import check_window_length
from sktime.utils.validation.forecasting import check_step_length


def _prepare_y_X(y, X):
    z = y.to_numpy()
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if X is not None:
        z = np.column_stack([z, X.to_numpy()])
    return z


def _prepare_fh(fh):
    assert fh.is_relative
    assert fh.is_all_out_of_sample()
    return fh.to_indexer().to_numpy()


def _sliding_window_transform(
    y, window_length, fh, X=None, scitype="tabular-regressor"
):
    """Transform time series data `z` with shape (n_timepoints, n_variables) into
    a 3d panel array via a sliding window of fixed length `window_length`.

    Returns
    -------
    yt : np.ndarray, shape = (n_timepoints - window_length, 1)
        Transformed target variable.
    Xt : np.ndarray, shape = (n_timepoints - window_length, n_variables, window_length)
        Transformed lagged values of target variable and exogenous variables,
        excluding contemporaneous values of exogenous variables.
    """
    # There are different ways to implement this transform. Pre-allocating an
    # array and filling it by iterating over the window length seems to be the most
    # efficient one.
    window_length = check_window_length(window_length)

    z = _prepare_y_X(y, X)
    n_timepoints, n_variables = z.shape

    fh = _prepare_fh(fh)
    fh_max = fh[-1]

    if window_length + fh_max >= n_timepoints:
        raise ValueError(
            "The `window_length` and `fh` are incompatible with the length of `y`"
        )

    # Get the effective window length accounting for the forecasting horizon.
    effective_window_length = window_length + fh_max

    # Pre-allocate array for sliding windows.
    Zt = np.zeros(
        (
            n_timepoints + effective_window_length,
            n_variables,
            effective_window_length + 1,
        )
    )

    # Transform data.
    for k in range(effective_window_length + 1):
        i = effective_window_length - k
        j = n_timepoints + effective_window_length - k
        Zt[i:j, :, k] = z

    # Truncate data, selecting only full windows, discarding incomplete ones.
    Zt = Zt[effective_window_length:-effective_window_length]

    # Return transformed feature and target variables separately. This excludes
    # contemporaneous values of the exogenous variables. Including them would lead to
    # unequal-length data, with more time points for exogenous series
    # than the target series, with is currently not supported.
    yt = Zt[:, 0, window_length + fh]
    Xt = Zt[:, :, :window_length]

    if scitype == "tabular-regressor":
        return yt, Xt.reshape(Xt.shape[0], -1)
    else:
        return yt, Xt


class _Reducer(_BaseWindowForecaster):
    """Base class for reducing forecasting to time series regression"""

    _required_parameters = ["estimator"]

    def __init__(self, estimator, window_length=10, step_length=1):
        super(_Reducer, self).__init__(window_length=window_length)
        self.estimator = estimator
        self.step_length = step_length
        self.step_length_ = None
        self._cv = None

    def fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self._set_y_X(y, X)
        self._set_fh(fh)

        self.step_length_ = check_step_length(self.step_length)
        self.window_length_ = check_window_length(self.window_length)

        self._fit(y, X)
        self._is_fitted = True
        return self

    def _fit(self, y, X):
        raise NotImplementedError("abstract method")

    def _is_predictable(self, last_window):
        """Helper function to check if we can make predictions from last
        window"""
        return (
            len(last_window) == self.window_length_
            and np.sum(np.isnan(last_window)) == 0
            and np.sum(np.isinf(last_window)) == 0
        )

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        # Note that we currently only support out-of-sample predictions. For the
        # direct and multioutput strategy, we need to check this already during fit,
        # as the fh is required for fitting.
        raise NotImplementedError(
            f"Generating in-sample predictions is not yet "
            f"implemented for {self.__class__.__name__}."
        )


class _DirectReducer(_RequiredForecastingHorizonMixin, _Reducer):
    strategy = "direct"

    def _transform(self, y, X=None):
        fh = self.fh.to_relative(self.cutoff)
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None):
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
        # We currently only support out-of-sample predictions. For the direct
        # strategy, we need to check this at the beginning of fit, as the fh is
        # required for fitting.
        if not self.fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError("In-sample predictions are not implemented.")

        yt, Xt = self._transform(y, X)

        # Iterate over forecasting horizon, fitting a separate estimator for each step.
        self.estimators_ = []
        for i in range(len(self.fh)):
            estimator = clone(self.estimator)
            estimator.fit(Xt, yt[:, [i]])
            self.estimators_.append(estimator)
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # Get last window of available data.
        y_last, X_last = self._get_last_window()

        # If we cannot generate a prediction from the available data, return nan.
        if not self._is_predictable(y_last):
            return self._predict_nan(fh)

        if self._X is None:
            n_columns = 1
        else:
            # X is ignored here, since we currently only look at lagged values for
            # exogenous variables and not contemporaneous ones.
            # X = X.to_numpy()
            n_columns = self._X.shape[1] + 1

        # Pre-allocate arrays.
        window_length = self.window_length_
        X_pred = np.zeros((1, n_columns, window_length))

        # Fill pre-allocated arrays with available data.
        X_pred[:, 0, :] = y_last
        if self._X is not None:
            X_pred[:, 1:, :] = X_last.T

        # We need to make sure that X has the same order as used in fit.
        if self._estimator_scitype == "tabular-regressor":
            X_pred = X_pred.reshape(1, -1)

        # Allocate array for predictions.
        y_pred = np.zeros(len(fh))

        # Iterate over estimators/forecast horizon
        for i, estimator in enumerate(self.estimators_):
            y_pred[i] = estimator.predict(X_pred)

        return y_pred


class _MultioutputReducer(_RequiredForecastingHorizonMixin, _Reducer):
    strategy = "multioutput"

    def _transform(self, y, X=None):
        fh = self.fh.to_relative(self.cutoff)
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None):
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
        # We currently only support out-of-sample predictions. For the direct
        # strategy, we need to check this at the beginning of fit, as the fh is
        # required for fitting.
        if not self.fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError("In-sample predictions are not implemented.")

        yt, Xt = self._transform(y, X)

        # Fit a multi-output estimator to the transformed data.
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xt, yt)
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # Get last window of available data.
        y_last, X_last = self._get_last_window()

        # If we cannot generate a prediction from the available data, return nan.
        if not self._is_predictable(y_last):
            return self._predict_nan(fh)

        if self._X is None:
            n_columns = 1
        else:
            # X is ignored here, since we currently only look at lagged values for
            # exogenous variables and not contemporaneous ones.
            # X = X.to_numpy()
            n_columns = self._X.shape[1] + 1

        # Pre-allocate arrays.
        window_length = self.window_length_
        X_pred = np.zeros((1, n_columns, window_length))

        # Fill pre-allocated arrays with available data.
        X_pred[:, 0, :] = y_last
        if self._X is not None:
            X_pred[:, 1:, :] = X_last.T

        # We need to make sure that X has the same order as used in fit.
        if self._estimator_scitype == "tabular-regressor":
            X_pred = X_pred.reshape(1, -1)

        # Iterate over estimators/forecast horizon
        y_pred = self.estimator_.predict(X_pred)
        return y_pred.ravel()


class _RecursiveReducer(_OptionalForecastingHorizonMixin, _Reducer):
    strategy = "recursive"

    def _transform(self, y, X=None):
        # For the recursive strategy, the forecasting horizon for the sliding-window
        # transform is simply a one-step ahead horizon, regardless of the horizon
        # used during prediction.
        fh = ForecastingHorizon([1])
        return _sliding_window_transform(
            y, self.window_length_, fh, X, scitype=self._estimator_scitype
        )

    def _fit(self, y, X):
        yt, Xt = self._transform(y, X)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xt, yt)
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        if self._X is not None and X is None:
            raise ValueError(
                "`X` must be passed to `predict` if `X` is given in `fit`."
            )

        # Get last window of available data.
        y_last, X_last = self._get_last_window()

        # If we cannot generate a prediction from the available data, return nan.
        if not self._is_predictable(y_last):
            return self._predict_nan(fh)

        # Pre-allocate arrays.
        if X is None:
            n_columns = 1
        else:
            X = X.to_numpy()
            n_columns = X.shape[1] + 1
        window_length = self.window_length_
        fh_max = fh.to_relative(self.cutoff)[-1]

        y_pred = np.zeros(fh_max)
        last = np.zeros((1, n_columns, window_length + fh_max))

        # Fill pre-allocated arrays with available data.
        last[:, 0, :window_length] = y_last
        if X is not None:
            last[:, 1:, :window_length] = X_last.T
            last[:, 1:, window_length:] = X.T

        # Recursively generate predictions by iterating over forecasting horizon.
        for i in range(fh_max):
            # Slice prediction window.
            X_pred = last[:, :, i : window_length + i]

            # Reshape data into tabular array.
            if self._estimator_scitype == "tabular-regressor":
                X_pred = X_pred.reshape(1, -1)

            # Generate predictions.
            y_pred[i] = self.estimator_.predict(X_pred)

            # Update last window with previous prediction.
            last[:, 0, window_length + i] = y_pred[i]

        # While the recursive strategy requires to generate predictions for all steps
        # until the furthest step in the forecasting horizon, we only return the
        # requested ones.
        fh_idx = fh.to_indexer(self.cutoff)
        return y_pred[fh_idx]


class DirectTabularRegressionForecaster(_DirectReducer):
    """
    Forecasting based on reduction to tabular regression with a direct
    reduction strategy.
    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon

    Parameters
    ----------
    estimator : sklearn estimator object
        Define the regression model type.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class MultioutputTabularRegressionForecaster(_MultioutputReducer):
    """
    Forecasting based on reduction to tabular regression with a multioutput
    reduction strategy.
    For the multioutput reduction strategy, a single forecaster is fitted
    simultaneously to all the future steps in the forecasting horizon

    Parameters
    ----------
    estimator : sklearn estimator object
        Define the regression model type.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class RecursiveTabularRegressionForecaster(_RecursiveReducer):
    """
    Forecasting based on reduction to tabular regression with a recursive
    reduction strategy.
    For the recursive reduction strategy, a single estimator is
    fit for a one-step-ahead forecasting horizon and then called
    iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : sklearn estimator object
        Define the regression model type.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class DirectTimeSeriesRegressionForecaster(_DirectReducer):
    """
    Forecasting based on reduction to time series regression with a direct
    reduction strategy.
    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon

    Parameters
    ----------
    estimator : sktime estimator object
        Define the type of time series regression model.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class MultioutputTimeSeriesRegressionForecaster(_MultioutputReducer):
    """
    Forecasting based on reduction to time series regression with a multioutput
    reduction strategy.
    For the recursive reduction strategy, a single estimator is
    fit for a one-step-ahead forecasting horizon and then called
    iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : sktime estimator object
        Define the type of time series regression model.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    pass
    _estimator_scitype = "time-series-regressor"


class RecursiveTimeSeriesRegressionForecaster(_RecursiveReducer):
    """
    Forecasting based on reduction to time series regression with a recursive
    reduction strategy.
    For the recursive reduction strategy, a single estimator is
    fit for a one-step-ahead forecasting horizon and then called
    iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : sktime estimator object
        Define the type of time series regression model.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    step_length : int, optional (default=1)
        The number of time steps taken at each step of the sliding window
        used to transform the series into a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


@deprecated("Please use `make_reduction` from `sktime.forecasting.compose` instead.")
def ReducedForecaster(
    estimator, scitype, strategy="recursive", window_length=10, step_length=1
):
    """
    Forecasting based on reduction

    When fitting, a rolling window approach is used to first transform the
    target series into panel data which is
    then used to train a estimator. During prediction, the last
    available data is used as input to the
    fitted estimators to make forecasts.

    Parameters
    ----------
    estimator : a estimator of type given by parameter scitype
    scitype : str
        Can be 'regressor' or 'ts-regressor'
    strategy : str {"direct", "recursive", "multioutput"}, optional
        Strategy to generate predictions
    window_length : int, optional (default=10)
    step_length : int, optional (default=1)

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (
    2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    if step_length != 1:
        raise ValueError("`step_length` is no longer supported.")
    return make_reduction(
        estimator, strategy=strategy, window_length=window_length, scitype=scitype
    )


@deprecated("Please use `make_reduction` from `sktime.forecasting.compose` instead.")
def ReducedRegressionForecaster(
    estimator, scitype, strategy="recursive", window_length=10, step_length=1
):
    """
    Forecasting based on reduction

    When fitting, a rolling window approach is used to first transform the
    target series into panel data which is
    then used to train a estimator. During prediction, the last
    available data is used as input to the
    fitted estimators to make forecasts.

    Parameters
    ----------
    estimator : a estimator of type given by parameter scitype
    scitype : str
        Can be 'regressor' or 'ts-regressor'
    strategy : str {"direct", "recursive", "multioutput"}, optional
        Strategy to generate predictions
    window_length : int, optional (default=10)
    step_length : int, optional (default=1)

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (
    2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    if step_length != 1:
        raise ValueError("`step_length` is no longer supported.")
    return make_reduction(
        estimator, strategy=strategy, window_length=window_length, scitype=scitype
    )


def make_reduction(
    estimator,
    strategy="recursive",
    window_length=10,
    scitype="infer",
):
    """
    Create a reduction forecaster based on a tabular or time series
    regression estimator.

    When fitting, a sliding window approach is used to first transform the
    target series into tabular or panel data, which is then used to train a
    estimator. During prediction, the last available data is used as input to the
    fitted estimators to generate forecasts.

    Parameters
    ----------
    estimator : an estimator instance
        Either a tabular regressor from scikit-learn or a time series regressor from
        sktime.
    strategy : str, optional (default="recursive")
        Must be one of "direct", "recursive", "multioutput". The strategy to apply
        estimator to forecasting task.
    window_length : int, optional (default=10)
        Window length used in sliding window transformation.
    scitype : str, optional (default="inferred")
        Must be one of "inferred", "tabular-regressor" or "time-series-regressor". If
        the scitype cannot be inferred, please specify it explicitly.

    Returns
    -------
    estimator : an Estimator instance
        A reduction forecaster

    References
    ----------
    ..[1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013).
      Machine Learning Strategies for Time Series Forecasting.
    """
    # We provide this function as a factory method for easier usage.
    strategy = _check_strategy(strategy)
    scitype = _check_scitype(scitype)

    if scitype == "infer":
        scitype = _infer_scitype(estimator)

    Forecaster = _get_forecaster(scitype, strategy)
    return Forecaster(estimator=estimator, window_length=window_length)


def _check_scitype(scitype):
    valid_scitypes = ("infer", "tabular-regressor", "time-series-regressor")
    if scitype not in valid_scitypes:
        raise ValueError(f"Unknown `scitype`, please use one of {valid_scitypes}.")

    return scitype


def _infer_scitype(estimator):
    # We can check if estimator is an instance of scikit-learn's RegressorMixin or
    # of sktime's BaseRegressor, otherwise we raise an error. Some time-series
    # regressor also inherit from scikit-learn classes, hence the order in which we
    # check matters.
    if isinstance(estimator, BaseRegressor):
        return "time-series-regressor"
    elif isinstance(estimator, RegressorMixin):
        return "tabular-regressor"
    else:
        raise ValueError(
            "The `scitype` of the given `estimator` cannot be inferred. "
            "Please specify the `scitype` explicitly."
        )


def _check_strategy(strategy):
    valid_strategies = ("direct", "recursive", "multioutput")
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown `strategy`, please use one of {valid_strategies}.")
    return strategy


def _get_forecaster(scitype, strategy):
    """Helper function to select forecaster for a given scientific type (
    scitype)
    and reduction strategy"""

    registry = {
        "tabular-regressor": {
            "direct": DirectTabularRegressionForecaster,
            "recursive": RecursiveTabularRegressionForecaster,
            "multioutput": MultioutputTabularRegressionForecaster,
        },
        "time-series-regressor": {
            "direct": DirectTimeSeriesRegressionForecaster,
            "recursive": RecursiveTimeSeriesRegressionForecaster,
            "multioutput": MultioutputTimeSeriesRegressionForecaster,
        },
    }
    return registry.get(scitype).get(strategy)
