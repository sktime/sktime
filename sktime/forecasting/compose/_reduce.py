#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Composition functionality for reduction approaches to forecasting."""

__author__ = [
    "Ayushmaan Seth",
    "Kavin Anand",
    "Luis Zugasti",
    "Lovkush Agarwal",
    "Markus Löning",
]

__all__ = [
    "make_reduction",
    "DirectTimeSeriesRegressionForecaster",
    "RecursiveTimeSeriesRegressionForecaster",
    "MultioutputTimeSeriesRegressionForecaster",
    "DirectTabularRegressionForecaster",
    "RecursiveTabularRegressionForecaster",
    "MultioutputTabularRegressionForecaster",
    "DirRecTabularRegressionForecaster",
    "DirRecTimeSeriesRegressionForecaster",
    "ReducedForecaster",
    "ReducedRegressionForecaster",
]

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.base import clone

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.regression.base import BaseRegressor
from sktime.utils._maint import deprecated
from sktime.utils.validation import check_window_length


def _concat_y_X(y, X):
    """Concatenate y and X prior to sliding-window transform."""
    z = y.to_numpy()
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if X is not None:
        z = np.column_stack([z, X.to_numpy()])
    return z


def _check_fh(fh):
    """Check fh prior to sliding-window transform."""
    assert fh.is_relative
    assert fh.is_all_out_of_sample()
    return fh.to_indexer().to_numpy()


def _sliding_window_transform(
    y, window_length, fh, X=None, scitype="tabular-regressor"
):
    """Transform time series data using sliding window.

    See `test_sliding_window_transform_explicit` in test_reduce.py for explicit
    example.

    Parameters
    ----------
    y : pd.Series
        Endogenous time series
    window_length : int
        Window length for transformed feature variables
    fh : ForecastingHorizon
        Forecasting horizon for transformed target variable
    X : pd.DataFrame, optional (default=None)
        Exogenous series.
    scitype : str {"tabular-regressor", "time-series-regressor"}, optional
        Scitype of estimator to use with transformed data.
        - If "tabular-regressor", returns X as tabular 2d array
        - If "time-series-regressor", returns X as panel 3d array

    Returns
    -------
    yt : np.ndarray, shape = (n_timepoints - window_length, 1)
        Transformed target variable.
    Xt : np.ndarray, shape = (n_timepoints - window_length, n_variables,
    window_length)
        Transformed lagged values of target variable and exogenous variables,
        excluding contemporaneous values.
    """
    # There are different ways to implement this transform. Pre-allocating an
    # array and filling it by iterating over the window length seems to be the most
    # efficient one.
    n_timepoints = y.shape[0]
    window_length = check_window_length(window_length, n_timepoints)

    z = _concat_y_X(y, X)
    n_timepoints, n_variables = z.shape

    fh = _check_fh(fh)
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
    # unequal-length data, with more time points for exogenous series than the target
    # series, which is currently not supported.
    yt = Zt[:, 0, window_length + fh]
    Xt = Zt[:, :, :window_length]

    # If the scitype is tabular regression, we have to convert X into a 2d array.
    if scitype == "tabular-regressor":
        return yt, Xt.reshape(Xt.shape[0], -1)
    else:
        return yt, Xt


class _Reducer(_BaseWindowForecaster):
    """Base class for reducing forecasting to regression."""

    _required_parameters = ["estimator"]

    def __init__(self, estimator, window_length=10):
        super(_Reducer, self).__init__(window_length=window_length)
        self.estimator = estimator
        self._cv = None

    def _is_predictable(self, last_window):
        """Check if we can make predictions from last window."""
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


class _DirectReducer(_Reducer):
    strategy = "direct"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        fh = self.fh.to_relative(self.cutoff)
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None, fh=None):
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
        self : Estimator
            An fitted instance of self.
        """
        # We currently only support out-of-sample predictions. For the direct
        # strategy, we need to check this at the beginning of fit, as the fh is
        # required for fitting.
        if not self.fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError("In-sample predictions are not implemented.")

        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        yt, Xt = self._transform(y, X)

        # Iterate over forecasting horizon, fitting a separate estimator for each step.
        self.estimators_ = []
        for i in range(len(self.fh)):
            estimator = clone(self.estimator)
            estimator.fit(Xt, yt[:, i])
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


class _MultioutputReducer(_Reducer):
    strategy = "multioutput"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        fh = self.fh.to_relative(self.cutoff)
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None, fh=None):
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
        # We currently only support out-of-sample predictions. For the direct
        # strategy, we need to check this at the beginning of fit, as the fh is
        # required for fitting.
        if not self.fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError("In-sample predictions are not implemented.")

        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

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


class _RecursiveReducer(_Reducer):
    strategy = "recursive"
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        # For the recursive strategy, the forecasting horizon for the sliding-window
        # transform is simply a one-step ahead horizon, regardless of the horizon
        # used during prediction.
        fh = ForecastingHorizon([1])
        return _sliding_window_transform(
            y, self.window_length_, fh, X, scitype=self._estimator_scitype
        )

    def _fit(self, y, X=None, fh=None):
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
        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        yt, Xt = self._transform(y, X)

        # Make sure yt is 1d array to avoid DataConversion warning from scikit-learn.
        yt = yt.ravel()

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


class _DirRecReducer(_Reducer):
    strategy = "dirrec"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
    }

    def _transform(self, y, X=None):
        # Note that the transform for dirrec is the same as in the direct
        # strategy.
        fh = self.fh.to_relative(self.cutoff)
        return _sliding_window_transform(
            y,
            window_length=self.window_length,
            fh=fh,
            X=X,
            scitype=self._estimator_scitype,
        )

    def _fit(self, y, X=None, fh=None):
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
        self : Estimator
            An fitted instance of self.
        """
        # Exogenous variables are not yet supported for the dirrec strategy.
        if X is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not yet support exogenous "
                f"variables `X`."
            )

        if len(self.fh.to_in_sample(self.cutoff)) > 0:
            raise NotImplementedError("In-sample predictions are not implemented")

        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )

        # Transform the data using sliding-window.
        yt, Xt = self._transform(y, X)

        # We cast the 2d tabular array into a 3d panel array to handle the data
        # consistently for the reduction to tabular and time-series regression.
        if self._estimator_scitype == "tabular-regressor":
            Xt = np.expand_dims(Xt, axis=1)

        # This only works without exogenous variables. To support exogenous
        # variables, we need additional values for X to fill the array
        # appropriately.
        X_full = np.concatenate([Xt, np.expand_dims(yt, axis=1)], axis=2)

        self.estimators_ = []
        n_timepoints = Xt.shape[2]

        for i in range(len(self.fh)):
            estimator = clone(self.estimator)

            # Slice data using expanding window.
            X_fit = X_full[:, :, : n_timepoints + i]

            # Convert to 2d tabular array for reduction to tabular regression.
            if self._estimator_scitype == "tabular-regressor":
                X_fit = X_fit.reshape(X_fit.shape[0], -1)

            estimator.fit(X_fit, yt[:, i])
            self.estimators_.append(estimator)

        self._is_fitted = True
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        # Exogenous variables are not yet support for the dirrec strategy.
        if X is not None:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not yet support exogenous "
                f"variables `X`."
            )

        # Get last window of available data.
        y_last, X_last = self._get_last_window()
        if not self._is_predictable(y_last):
            return self._predict_nan(fh)

        window_length = self.window_length_

        # Pre-allocated arrays.
        # We set `n_columns` here to 1, because exogenous variables
        # are not yet supported.
        n_columns = 1
        X_full = np.zeros((1, n_columns, window_length + len(self.fh)))
        X_full[:, 0, :window_length] = y_last

        y_pred = np.zeros(len(fh))

        for i in range(len(self.fh)):

            # Slice data using expanding window.
            X_pred = X_full[:, :, : window_length + i]

            if self._estimator_scitype == "tabular-regressor":
                X_pred = X_pred.reshape(1, -1)

            y_pred[i] = self.estimators_[i].predict(X_pred)

            # Update the last window with previously predicted value.
            X_full[:, :, window_length + i] = y_pred[i]

        return y_pred


class DirectTabularRegressionForecaster(_DirectReducer):
    """
    Direct reduction from forecasting to tabular regression.

    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class MultioutputTabularRegressionForecaster(_MultioutputReducer):
    """
    Multioutput reduction from forecasting to tabular regression.

    For the multioutput strategy, a single estimator capable of handling multioutput
    targets is fitted to all the future steps in the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class RecursiveTabularRegressionForecaster(_RecursiveReducer):
    """
    Recursive reduction from forecasting to tabular regression.

    For the recursive strategy, a single estimator is fit for a one-step-ahead
    forecasting horizon and then called iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : Estimator
        A tabular regression estimator as provided by scikit-learn.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "tabular-regressor"


class DirRecTabularRegressionForecaster(_DirRecReducer):
    """
    Dir-rec reduction from forecasting to tabular regression.

    For the hybrid dir-rec strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon and then
    the previous forecasting horizon is added as an input
    for training the next forecaster, following the recusrive
    strategy.

    Parameters
    ----------
    estimator : sklearn estimator object
        Tabular regressor.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    """

    _estimator_scitype = "tabular-regressor"


class DirectTimeSeriesRegressionForecaster(_DirectReducer):
    """
    Direct reduction from forecasting to time-series regression.

    For the direct reduction strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class MultioutputTimeSeriesRegressionForecaster(_MultioutputReducer):
    """
    Multioutput reduction from forecasting to time series regression.

    For the multioutput strategy, a single estimator capable of handling multioutput
    targets is fitted to all the future steps in the forecasting horizon.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class RecursiveTimeSeriesRegressionForecaster(_RecursiveReducer):
    """
    Recursive reduction from forecasting to time series regression.

    For the recursive strategy, a single estimator is fit for a one-step-ahead
    forecasting horizon and then called iteratively to predict multiple steps ahead.

    Parameters
    ----------
    estimator : Estimator
        A time-series regression estimator as provided by sktime.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix.
    """

    _estimator_scitype = "time-series-regressor"


class DirRecTimeSeriesRegressionForecaster(_DirRecReducer):
    """
    Dir-rec reduction from forecasting to time-series regression.

    For the hybrid dir-rec strategy, a separate forecaster is fitted
    for each step ahead of the forecasting horizon and then
    the previous forecasting horizon is added as an input
    for training the next forecaster, following the recusrive
    strategy.

    Parameters
    ----------
    estimator : sktime estimator object
        Time-series regressor.
    window_length : int, optional (default=10)
        The length of the sliding window used to transform the series into
        a tabular matrix
    """

    _estimator_scitype = "time-series-regressor"


@deprecated("Please use `make_reduction` from `sktime.forecasting.compose` instead.")
def ReducedForecaster(
    estimator, scitype="infer", strategy="recursive", window_length=10, step_length=1
):
    """
    Reduction from forecasting to tabular or time series regression.

    During fitting, a sliding-window approach is used to first transform the
    time series into tabular or panel data, which is then used to fit a tabular or
    time-series regression estimator. During prediction, the last available data is
    used as input to the fitted regression estimator to generate forecasts.

    Parameters
    ----------
    estimator : Estimator
        A estimator of type given by parameter scitype
    scitype : str {"infer", "tabular-regressor", "time-series-regressor"}
        Scitype of estimator.
    strategy : str {"recursive", "direct", "multioutput"}, optional
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
        raise ValueError(
            "`step_length` values different from 1 are no longer supported."
        )
    return make_reduction(
        estimator, strategy=strategy, window_length=window_length, scitype=scitype
    )


@deprecated("Please use `make_reduction` from `sktime.forecasting.compose` instead.")
def ReducedRegressionForecaster(
    estimator, scitype, strategy="recursive", window_length=10, step_length=1
):
    """
    Reduction from forecasting to tabular or time series regression.

    During fitting, a sliding-window approach is used to first transform the
    time series into tabular or panel data, which is then used to fit a tabular or
    time-series regression estimator. During prediction, the last available data is
    used as input to the fitted regression estimator to generate forecasts.

    Parameters
    ----------
    estimator : a estimator of type given by parameter scitype
    scitype : str {"infer", "tabular-regressor", "time-series-regressor"}
        Scitype of estimator.
    strategy : str {"recursive", "direct", "multioutput"}, optional
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
        raise ValueError(
            "`step_length` values different from 1 are no longer " "supported."
        )
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
    Make forecaster based on reduction to tabular or time-series regression.

    During fitting, a sliding-window approach is used to first transform the
    time series into tabular or panel data, which is then used to fit a tabular or
    time-series regression estimator. During prediction, the last available data is
    used as input to the fitted regression estimator to generate forecasts.

    Parameters
    ----------
    estimator : an estimator instance
        Either a tabular regressor from scikit-learn or a time series regressor from
        sktime.
    strategy : str, optional (default="recursive")
        The strategy to generate forecasts. Must be one of "direct", "recursive" or
        "multioutput".
    window_length : int, optional (default=10)
        Window length used in sliding window transformation.
    scitype : str, optional (default="infer")
        Must be one of "infer", "tabular-regressor" or "time-series-regressor". If
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
    # We provide this function as a factory method for user convenience.
    strategy = _check_strategy(strategy)
    scitype = _check_scitype(scitype)

    if scitype == "infer":
        scitype = _infer_scitype(estimator)

    Forecaster = _get_forecaster(scitype, strategy)
    return Forecaster(estimator=estimator, window_length=window_length)


def _check_scitype(scitype):
    valid_scitypes = ("infer", "tabular-regressor", "time-series-regressor")
    if scitype not in valid_scitypes:
        raise ValueError(
            f"Invalid `scitype`. `scitype` must be one of:"
            f" {valid_scitypes}, but found: {scitype}."
        )
    return scitype


def _infer_scitype(estimator):
    # We can check if estimator is an instance of scikit-learn's RegressorMixin or
    # of sktime's BaseRegressor, otherwise we raise an error. Note that some time-series
    # regressor also inherit from scikit-learn classes, hence the order in which we
    # check matters and we first need to check for BaseRegressor.
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
    valid_strategies = ("direct", "recursive", "multioutput", "dirrec")
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid `strategy`. `strategy` must be one of :"
            f" {valid_strategies}, but found: {strategy}."
        )
    return strategy


def _get_forecaster(scitype, strategy):
    """Select forecaster for a given scientific type and reduction strategy."""
    registry = {
        "tabular-regressor": {
            "direct": DirectTabularRegressionForecaster,
            "recursive": RecursiveTabularRegressionForecaster,
            "multioutput": MultioutputTabularRegressionForecaster,
            "dirrec": DirRecTabularRegressionForecaster,
        },
        "time-series-regressor": {
            "direct": DirectTimeSeriesRegressionForecaster,
            "recursive": RecursiveTimeSeriesRegressionForecaster,
            "multioutput": MultioutputTimeSeriesRegressionForecaster,
            "dirrec": DirRecTimeSeriesRegressionForecaster,
        },
    }
    return registry[scitype][strategy]
