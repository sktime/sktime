#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Composition functionality for reduction approaches to forecasting."""

__author__ = [
    "mloning",
    "AyushmaanSeth",
    "Kavin Anand",
    "Luis Zugasti",
    "Lovkush-A",
    "fkiraly",
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
    "DirectReductionForecaster",
]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.multioutput import MultiOutputRegressor

from sktime.datatypes._utilities import get_time_index
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._fh import _index_range
from sktime.forecasting.base._sktime import _BaseWindowForecaster
from sktime.regression.base import BaseRegressor
from sktime.transformations.compose import FeatureUnion
from sktime.utils.datetime import _shift
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
    y, window_length, fh, X=None, transformers=None, scitype="tabular-regressor"
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
    transformers: *experimental*
        A suitable transformer that allows for using an en-bloc approach with
        make_reduction. This means that instead of using the raw past observations of
        y across the window length, suitable features will be generated directly from
        the past raw observations. Currently only supports WindowSummarizer to generate
        e.g. the mean of the past 7 observations.
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

    ts_index = get_time_index(y)
    n_timepoints = ts_index.shape[0]
    window_length = check_window_length(window_length, n_timepoints)

    if transformers is not None:
        if len(transformers) == 1:
            tf_fit = transformers[0].fit(y)
        else:
            feat = [("trafo_" + str(index), i) for index, i in enumerate(transformers)]
            tf_fit = FeatureUnion(feat).fit(y)
        X_from_y = tf_fit.transform(y)

        X_from_y_cut = _cut_tail(X_from_y, n_tail=n_timepoints - window_length)
        yt = _cut_tail(y, n_tail=n_timepoints - window_length)

        if X is not None:
            X_cut = _cut_tail(X, n_tail=n_timepoints - window_length)
            Xt = pd.concat([X_from_y_cut, X_cut], axis=1)
        else:
            Xt = X_from_y_cut
    else:
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

        # Return transformed feature and target variables separately. This
        # excludes contemporaneous values of the exogenous variables. Including them
        # would lead to unequal-length data, with more time points for
        # exogenous series than the target series, which is currently not supported.
        yt = Zt[:, 0, window_length + fh]
        Xt = Zt[:, :, :window_length]
    # Pre-allocate array for sliding windows.
    # If the scitype is tabular regression, we have to convert X into a 2d array.
    if scitype == "tabular-regressor":
        if transformers is not None:
            return yt, Xt
        else:
            return yt, Xt.reshape(Xt.shape[0], -1)
    else:
        return yt, Xt


class _Reducer(_BaseWindowForecaster):
    """Base class for reducing forecasting to regression."""

    _tags = {
        "ignores-exogeneous-X": False,  # reduction uses X in non-trivial way
        "handles-missing-data": True,
    }

    def __init__(self, estimator, window_length=10, transformers=None):
        super(_Reducer, self).__init__(window_length=window_length)
        self.transformers = transformers
        self.transformers_ = None
        self.estimator = estimator
        self._cv = None

        # it seems that the sklearn tags are not fully reliable
        # see discussion in PR #3405 and issue #3402
        # therefore this is commented out until sktime and sklearn are better aligned
        # self.set_tags(**{"handles-missing-data": estimator._get_tags()["allow_nan"]})

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline

        from sktime.transformations.panel.reduce import Tabularizer

        # naming convention is as follows:
        #   reducers with Tabular take an sklearn estimator, e.g., LinearRegressor
        #   reducers with TimeSeries take an sktime supervised estimator
        #       e.g., pipeline of Tabularizer and Linear Regression
        # which of these is the case, we check by checking substring in the class name
        est = LinearRegression()
        if "TimeSeries" in cls.__name__:
            est = make_pipeline(Tabularizer(), est)

        params = {"estimator": est, "window_length": 3}
        return params


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
            transformers=self.transformers_,
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
        """Fit to training data.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series or pd.DataFrame
        """
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
        """Predict to training data.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series or pd.DataFrame
        """
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

    def _transform(self, y, X=None):
        # For the recursive strategy, the forecasting horizon for the sliding-window
        # transform is simply a one-step ahead horizon, regardless of the horizon
        # used during prediction.
        fh = ForecastingHorizon([1])
        return _sliding_window_transform(
            y,
            self.window_length_,
            fh,
            X=X,
            transformers=self.transformers_,
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
        if self.window_length is not None and self.transformers is not None:
            raise ValueError(
                "Transformers provided, suggesting en-bloc approach"
                + " to derive reduction features. Window length will be"
                + " inferred, please set to None"
            )
        self.window_length_ = check_window_length(
            self.window_length, n_timepoints=len(y)
        )
        if self.transformers is not None:
            self.transformers_ = clone(self.transformers)
            self.transformers = clone(self.transformers)

        if self.window_length is None:
            trafo = self.transformers_
            fit_trafo = [i.fit(y) for i in trafo]
            ts = [i.truncate_start for i in fit_trafo if hasattr(i, "truncate_start")]
            if len(ts) > 0:
                self.window_length_ = max(ts)
            else:
                raise ValueError(
                    "Reduce must either have window length as argument"
                    + "or needs to have it passed by transformer via"
                    + "truncate_start"
                )
        yt, Xt = self._transform(y, X)

        # Make sure yt is 1d array to avoid DataConversion warning from scikit-learn.
        if self.transformers is not None:
            yt = yt.to_numpy().ravel()
        else:
            yt = yt.ravel()

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(Xt, yt)
        return self

    def _get_shifted_window(self, shift=0, y_update=None, X_update=None):
        """Get the start and end points of a shifted window.

        In recursive forecasting, the time based features need to be recalculated for
        every time step that is forecast. This is done in an iterative fashion over
        every forecasting horizon step. Shift specifies the timestemp over which the
        iteration is done, i.e. a shift of 0 will get a window between window_length
        steps in the past and t=0, shift = 1 will be window_length - 1 steps in the past
        and t= 1 etc- up to the forecasting horizon.

        Will also apply any transformers passed to the recursive reducer to y. This en
        bloc approach of directly applying the transformers is more efficient than
        creating all lags first across the window and then applying the transformers
        to the lagged data.

        Please see below a graphical representation of the logic using the following
        symbols:

        ``z`` = first observation to forecast.
        Not part of the window.
        ``*`` = (other) time stamps in the window which is summarized
        ``x`` = observations, past or future, not part of the window

        For`window_length = 7` and `fh = [3]` we get the following windows

        `shift = 0`
        |--------------------------- |
        | x x x x * * * * * * * z x x|
        |----------------------------|

        `shift = 1`
        |--------------------------- |
        | x x x x x * * * * * * * z x|
        |----------------------------|

        `shift = 2`
        |--------------------------- |
        | x x x x x x * * * * * * * z|
        |----------------------------|

        Parameters
        ----------
        shift : integer
            this will be correspond to the shift of the window_length into the future
        y_update : a pandas Series or Dataframe
            y values that were obtained in the recursive fashion.
        X_update : a pandas Series or Dataframe
            X values also need to be cut based on the into windows, see above.

        Returns
        -------
        y, X: A pandas dataframe or series
            contains the y and X data prepared for the respective windows, see above.

        """
        # shift and cutoff determine start and end of the window, respectively.
        start = _shift(self._cutoff, by=shift - self.window_length_)
        cutoff = _shift(self._cutoff, by=shift)

        if self.transformers_ is not None:
            # Get the last window of the endogenous variable.
            # If X is given, also get the last window of the exogenous variables.
            # relative _int will give the integer indices of the window defined above
            relative_int = pd.Index(list(map(int, range(-self.window_length_, 1))))
            # index_range will give the same indices,
            # but using the date format of cutoff
            index_range = _index_range(relative_int, cutoff)

            # y_raw is defined solely for the purpose of deriving a dataframe
            # window_length forecasting steps into the past in order to calculate the
            # new X from y features based on the transformer provided
            y_raw = _create_fcst_df(index_range, self._y)

            # The y_raw dataframe will contain historical and / or recursively
            # forecast value to calculate the new X features.

            # Historical values are passed here for all time steps of y_raw that lie in
            # the past .
            y_raw.update(self._y)

            # Forecast values are passed here for all time steps of y_raw that lie in
            # the future and were forecast in previous iterations.
            if y_update is not None:
                y_raw.update(y_update)

            # After filling the empty y_raw frame with historic / forecast values
            # X from y features can be calculated based on the passed transformer.
            if len(self.transformers_) == 1:
                X_from_y = self.transformers_[0].fit_transform(y_raw)
            else:
                ref = self.transformers_
                feat = [("trafo_" + str(index), i) for index, i in enumerate(ref)]
                X_from_y = FeatureUnion(feat).fit_transform(y_raw)
            # We are only interested in the last observations, since only that one
            # contains relevant value. In recursive forecasting, only one observations
            # can be forecast at a time.
            X_from_y_cut = _cut_tail(X_from_y)

            # X_from_y_cut is added to X dataframe (unlike y_raw, the X dataframe can
            # directly be created with one observation from the start,
            # since no features need to be calculated).
            if self._X is not None:
                X = _create_fcst_df([index_range[-1]], self._X)
                X.update(self._X)
                if X_update is not None:
                    X.update(X_update)
                X_cut = _cut_tail(X)
                X = pd.concat([X_from_y_cut, X_cut], axis=1)
            else:
                X = X_from_y_cut
            y = _cut_tail(y_raw)
        else:
            # Get the last window of the endogenous variable.
            y = self._y.loc[start:cutoff].to_numpy()
            # If X is given, also get the last window of the exogenous variables.
            X = self._X.loc[start:cutoff].to_numpy() if self._X is not None else None
        return y, X

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """.

        In recursive reduction, iteration must be done over the
        entire forecasting horizon. Specifically, when transformers are
        applied to y that generate features in X, forecasting must be done step by
        step to integrate the latest prediction of for the new set of features in
        X derived from that y.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_return = pd.Series or pd.DataFrame
        """
        if self._X is not None and X is None:
            raise ValueError(
                "`X` must be passed to `predict` if `X` is given in `fit`."
            )

        # Get last window of available data.
        if self.transformers_ is not None:
            y_last, X_last = self._get_shifted_window()
        else:
            y_last, X_last = self._get_last_window()

        # If we cannot generate a prediction from the available data, return nan.
        if self.transformers_ is None:
            if not self._is_predictable(y_last):
                return self._predict_nan(fh)
        else:
            ys = np.array(y_last)
            if not np.sum(np.isnan(ys)) == 0 and np.sum(np.isinf(ys)) == 0:
                return self._predict_nan(fh)

        if self.transformers_ is not None:
            fh_max = fh.to_relative(self.cutoff)[-1]
            relative = pd.Index(list(map(int, range(1, fh_max + 1))))
            index_range = _index_range(relative, self.cutoff)

            y_pred = _create_fcst_df(index_range, self._y)

            for i in range(fh_max):
                # Generate predictions.
                y_pred_vector = self.estimator_.predict(X_last)
                y_pred_curr = _create_fcst_df(
                    [index_range[i]], self._y, fill=y_pred_vector
                )
                y_pred.update(y_pred_curr)

                # # Update last window with previous prediction.
                if i + 1 != fh_max:
                    y_last, X_last = self._get_shifted_window(
                        y_update=y_pred, X_update=X, shift=i + 1
                    )

        else:
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
                last[:, 1:, window_length:] = X.iloc[
                    -(last.shape[2] - window_length) :, :
                ].T

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

        if isinstance(self._y.index, pd.MultiIndex):
            yi_grp = self._y.index.names[0:-1]
            y_return = y_pred.groupby(yi_grp, as_index=False).nth(fh_idx.to_list())
        elif isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            y_return = y_pred.iloc[fh_idx]
            if hasattr(y_return.index, "freq"):
                if y_return.index.freq != y_pred.index.freq:
                    y_return.index.freq = None
        else:
            y_return = y_pred[fh_idx]

        return y_return


class _DirRecReducer(_Reducer):
    strategy = "dirrec"
    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
        "ignores-exogeneous-X": True,
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
        # todo: logic for X below is broken. Escape X until fixed.
        if X is not None:
            X = None

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
        return self

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Fit to training data.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series or pd.DataFrame
        """
        # Exogenous variables are not yet support for the dirrec strategy.
        # todo: implement this. For now, we escape.
        if X is not None:
            X = None

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
    """Direct reduction from forecasting to tabular regression.

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
    """Multioutput reduction from forecasting to tabular regression.

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
    """Recursive reduction from forecasting to tabular regression.

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

    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        # hierarchical data types are commented out until #3316 is fixed
        # "y_inner_mtype": ["pd.Series", "pd-multiindex", "pd_multiindex_hier"],
        # "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
    }

    _estimator_scitype = "tabular-regressor"


class DirRecTabularRegressionForecaster(_DirRecReducer):
    """Dir-rec reduction from forecasting to tabular regression.

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
    """Direct reduction from forecasting to time-series regression.

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
    """Multioutput reduction from forecasting to time series regression.

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
    """Recursive reduction from forecasting to time series regression.

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

    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }

    _estimator_scitype = "time-series-regressor"


class DirRecTimeSeriesRegressionForecaster(_DirRecReducer):
    """Dir-rec reduction from forecasting to time-series regression.

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


def make_reduction(
    estimator,
    strategy="recursive",
    window_length=10,
    scitype="infer",
    transformers=None,
):
    """Make forecaster based on reduction to tabular or time-series regression.

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
        See :term:`scitype`.

    Returns
    -------
    estimator : an Estimator instance
        A reduction forecaster

    Examples
    --------
    >>> from sktime.forecasting.compose import make_reduction
    >>> from sktime.datasets import load_airline
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> y = load_airline()
    >>> regressor = GradientBoostingRegressor()
    >>> forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
    >>> forecaster.fit(y)
    RecursiveTabularRegressionForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])

    References
    ----------
    .. [1] Bontempi, Gianluca & Ben Taieb, Souhaib & Le Borgne, Yann-Aël. (2013).
        Machine Learning Strategies for Time Series Forecasting.
    """
    # We provide this function as a factory method for user convenience.
    strategy = _check_strategy(strategy)
    scitype = _check_scitype(scitype)

    if scitype == "infer":
        scitype = _infer_scitype(estimator)

    Forecaster = _get_forecaster(scitype, strategy)
    return Forecaster(
        estimator=estimator, window_length=window_length, transformers=transformers
    )


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


def _cut_tail(X, n_tail=1):
    """Cut input at tail, supports grouping."""
    if isinstance(X.index, pd.MultiIndex):
        Xi_grp = X.index.names[0:-1]
        X = X.groupby(Xi_grp, as_index=False).tail(n_tail)
    else:
        X = X.tail(n_tail)
    return X


def _create_fcst_df(target_date, origin_df, fill=None):
    """Create an empty multiindex dataframe from origin dataframe.

    In recursive forecasting, a new dataframe needs to be created that collects
    all forecasting steps (even for forecasting horizons other than those of interests).
    For example for fh =[1,2,12] we need the whole forecasting horizons from 1 to 12.

    Parameters
    ----------
    target_date : a list of dates
        this will be correspond to the new timepoints index to be created in the
        forecasting dataframe
    origin_df : a pandas Series or Dataframe
        the origin_df corresponds to the dataframe with the historic data. Useful
        information inferred from that dataframe is the index of the historic dataframe
        as well as the names of the original columns and the type of the object
        (dataframe or series)
    fill : a numpy.ndarray (optional)
        Corresponds to a numpy array of values that is used to fill up the dataframe.
        Useful when forecasts are returned from a forecasting models that discards
        the hierarchical structure of the input pandas dataframe

    Returns
    -------
    A pandas dataframe or series
    """
    oi = origin_df.index
    if not isinstance(oi, pd.MultiIndex):
        if isinstance(origin_df, pd.Series):
            if fill is None:
                template = pd.Series(np.zeros(len(target_date)), index=target_date)
            else:
                template = pd.Series(fill, index=target_date)
            template.name = origin_df.name
            return template
        else:
            if fill is None:
                template = pd.DataFrame(
                    np.zeros((len(target_date), len(origin_df.columns))),
                    index=target_date,
                    columns=origin_df.columns.to_list(),
                )
            else:
                template = pd.DataFrame(
                    fill, index=target_date, columns=origin_df.columns.to_list()
                )
            return template
    else:
        idx = origin_df.index.to_frame(index=False)
        instance_names = idx.columns[0:-1].to_list()
        time_names = idx.columns[-1]
        idx = idx[instance_names].drop_duplicates()

        timeframe = pd.DataFrame(target_date, columns=[time_names])

        target_frame = idx.merge(timeframe, how="cross")
        freq_inferred = target_date[0].freq
        mi = (
            target_frame.groupby(instance_names, as_index=True)
            .apply(
                lambda df: df.drop(instance_names, axis=1)
                .set_index(time_names)
                .asfreq(freq_inferred)
            )
            .index
        )

        if fill is None:
            template = pd.DataFrame(
                np.zeros((len(target_date) * idx.shape[0], len(origin_df.columns))),
                index=mi,
                columns=origin_df.columns.to_list(),
            )
        else:
            template = pd.DataFrame(
                fill,
                index=mi,
                columns=origin_df.columns.to_list(),
            )

        template = template.astype(origin_df.dtypes.to_dict())
        return template


def _coerce_col_str(X):
    """Coerce columns to string, to satisfy sklearn convention."""
    X.columns = [str(x) for x in X.columns]
    return X


def slice_at_ix(df, ix):
    """Slice pd.DataFrame at one index value, valid for simple Index and MultiIndex.

    Parameters
    ----------
    df : pd.DataFrame
    ix : pandas compatible index value

    Returns
    -------
    pd.DataFrame, row(s) of df, sliced at last (-1 st) level of df being equal to ix
        all index levels are retained in the return, none are dropped
    """
    if isinstance(df.index, pd.MultiIndex):
        return df.xs(ix, level=-1, axis=0, drop_level=False)
    else:
        return df.loc[[ix]]


class _ReducerMixin:
    """Common utilities for reducers."""

    def _get_expected_pred_idx(self, fh):
        """Construct DataFrame Index expected in y_pred, return of _predict.

        Parameters
        ----------
        fh : ForecastingHorizon, fh of self

        Returns
        -------
        fh_idx : pd.Index, expected index of y_pred returned by _predict
            CAVEAT: sorted by index level -1, since reduction is applied by fh
        """
        fh_idx = pd.Index(fh.to_absolute(self.cutoff))
        y_index = self._y.index

        if isinstance(y_index, pd.MultiIndex):
            y_inst_idx = y_index.droplevel(-1).unique()
            if isinstance(y_inst_idx, pd.MultiIndex):
                fh_idx = pd.Index([x + (y,) for y in fh_idx for x in y_inst_idx])
            else:
                fh_idx = pd.Index([(x, y) for y in fh_idx for x in y_inst_idx])

        return fh_idx


class DirectReductionForecaster(BaseForecaster, _ReducerMixin):
    """Direct reduction forecaster, incl single-output, multi-output, exogeneous Dir.

    Implements direct reduction, of forecasting to tabular regression.

    For no `X`, defaults to DirMO (direct multioutput) for `X_treatment = "concurrent"`,
    and simple direct (direct single-output) for `X_treatment = "shifted"`.

    Direct single-output with concurrent `X` behaviour can be configured
    by passing a single-output `scikit-learn` compatible transformer.

    Algorithm details:

    In `fit`, given endogeneous time series `y` and possibly exogeneous `X`:
        fits `estimator` to feature-label pairs as defined as follows.
    if `X_treatment = "concurrent":
        features = `y(t)`, `y(t-1)`, ..., `y(t-window_size)`, if provided: `X(t+h)`
        labels = `y(t+h)` for `h` in the forecasting horizon
        ranging over all `t` where the above have been observed (are in the index)
        for each `h` in the forecasting horizon (separate estimator fitted per `h`)
    if `X_treatment = "shifted":
        features = `y(t)`, `y(t-1)`, ..., `y(t-window_size)`, if provided: `X(t)`
        labels = `y(t+h_1)`, ..., `y(t+h_k)` for `h_j` in the forecasting horizon
        ranging over all `t` where the above have been observed (are in the index)
        estimator is fitted as a multi-output estimator (for all  `h_j` simultaneously)

    In `predict`, given possibly exogeneous `X`, at cutoff time `c`,
    if `X_treatment = "concurrent":
        applies fitted estimators' predict to
        feature = `y(c)`, `y(c-1)`, ..., `y(c-window_size)`, if provided: `X(c+h)`
        to obtain a prediction for `y(c+h)`, for each `h` in the forecasting horizon
    if `X_treatment = "shifted":
        applies fitted estimator's predict to
        features = `y(c)`, `y(c-1)`, ..., `y(c-window_size)`, if provided: `X(t)`
        to obtain prediction for `y(c+h_1)`, ..., `y(c+h_k)` for `h_j` in forec. horizon

    Parameters
    ----------
    estimator : sklearn regressor, must be compatible with sklearn interface
        tabular regression algorithm used in reduction algorithm
    window_length : int, optional, default=10
        window length used in the reduction algorithm
    transformers : currently not used
    X_treatment : str, optional, one of "concurrent" (default) or "shifted"
        determines the timestamps of X from which y(t+h) is predicted, for horizon h
        "concurrent": y(t+h) is predicted from lagged y, and X(t+h), for all h in fh
            in particular, if no y-lags are specified, y(t+h) is predicted from X(t)
        "shifted": y(t+h) is predicted from lagged y, and X(t), for all h in fh
            in particular, if no y-lags are specified, y(t+h) is predicted from X(t+h)
    impute : str or None, optional, method string passed to Imputer
        default="bfill", admissible strings are of Imputer.method parameter, see there
        if None, no imputation is done when applying Lag transformer to obtain inner X
    pooling : str, one of ["local", "global", "panel"], optional, default="local"
        level on which data are pooled to fit the supervised regression model
        "local" = unit/instance level, one reduced model per lowest hierarchy level
        "global" = top level, one reduced model overall, on pooled data ignoring levels
        "panel" = second lowest level, one reduced model per panel level (-2)
        if there are 2 or less levels, "global" and "panel" result in the same
        if there is only 1 level (single time series), all three settings agree
    """

    _tags = {
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
        "ignores-exogeneous-X": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        transformers=None,
        X_treatment="concurrent",
        impute_method="bfill",
        pooling="local",
    ):
        self.window_length = window_length
        self.transformers = transformers
        self.transformers_ = None
        self.estimator = estimator
        self.X_treatment = X_treatment
        self.impute_method = impute_method
        self.pooling = pooling
        self._lags = list(range(window_length))
        super(DirectReductionForecaster, self).__init__()

        warn(
            "DirectReductionForecaster is experimental, and interfaces may change. "
            "user feedback is appreciated in issue #3224 here: "
            "https://github.com/sktime/sktime/issues/3224"
        )

        if pooling == "local":
            mtypes = "pd.DataFrame"
        elif pooling == "global":
            mtypes = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        elif pooling == "panel":
            mtypes = ["pd.DataFrame", "pd-multiindex"]
        else:
            raise ValueError(
                "pooling in DirectReductionForecaster must be one of"
                ' "local", "global", "panel", '
                f"but found {pooling}"
            )
        self.set_tags(**{"X_inner_mtype": mtypes})
        self.set_tags(**{"y_inner_mtype": mtypes})

        # it seems that the sklearn tags are not fully reliable
        # see discussion in PR #3405 and issue #3402
        # therefore this is commented out until sktime and sklearn are better aligned
        # self.set_tags(**{"handles-missing-data": estimator._get_tags()["allow_nan"]})

    def _fit(self, y, X=None, fh=None):
        """Fit dispatcher based on X_treatment."""
        methodname = f"_fit_{self.X_treatment}"
        return getattr(self, methodname)(y=y, X=X, fh=fh)

    def _predict(self, X=None, fh=None):
        """Predict dispatcher based on X_treatment."""
        methodname = f"_predict_{self.X_treatment}"
        return getattr(self, methodname)(X=X, fh=fh)

    def _fit_shifted(self, y, X=None, fh=None):
        """Fit to training data."""
        from sktime.transformations.series.impute import Imputer
        from sktime.transformations.series.lag import Lag

        impute_method = self.impute_method

        # lagger_y_to_X_ will lag y to obtain the sklearn X
        lags = self._lags
        lagger_y_to_X = Lag(lags=lags, index_out="extend")
        if impute_method is not None:
            lagger_y_to_X = lagger_y_to_X * Imputer(method=impute_method)
        self.lagger_y_to_X_ = lagger_y_to_X

        # lagger_y_to_y_ will lag y to obtain the sklearn y
        fh_rel = fh.to_relative(self.cutoff)
        y_lags = list(fh_rel)
        y_lags = [-x for x in y_lags]
        lagger_y_to_y = Lag(lags=y_lags, index_out="original")
        self.lagger_y_to_y_ = lagger_y_to_y

        yt = lagger_y_to_y.fit_transform(y)
        y_notna = yt.notnull().all(axis=1)
        y_notna_idx = y_notna.index[y_notna]

        # we now check whether the set of full lags is empty
        # if yes, we set a flag, since we cannot fit the reducer
        # instead, later, we return a dummy prediction
        if len(y_notna_idx) == 0:
            self.empty_lags_ = True
            self.dummy_value_ = y.mean()
            return self
        else:
            self.empty_lags_ = False

        yt = yt.loc[y_notna_idx]
        Xt = lagger_y_to_X.fit_transform(y).loc[y_notna_idx]

        if X is not None:
            Xt = pd.concat([X.loc[y_notna_idx], Xt], axis=1)

        Xt = _coerce_col_str(Xt)
        yt = _coerce_col_str(yt)

        estimator = clone(self.estimator)
        if not estimator._get_tags()["multioutput"]:
            estimator = MultiOutputRegressor(estimator)
        estimator.fit(Xt, yt)
        self.estimator_ = estimator

        return self

    def _predict_shifted(self, fh=None, X=None):
        """Predict core logic."""
        y_cols = self._y.columns
        fh_idx = self._get_expected_pred_idx(fh=fh)

        if self.empty_lags_:
            ret = pd.DataFrame(index=fh_idx, columns=y_cols)
            for i in ret.index:
                ret.loc[i] = self.dummy_value_
            return ret

        lagger_y_to_X = self.lagger_y_to_X_

        Xt_lastrow = slice_at_ix(lagger_y_to_X.transform(self._y), self.cutoff)
        if self._X is not None:
            exog_X_lastrow = slice_at_ix(self._X, self.cutoff)
            Xt_lastrow = pd.concat([exog_X_lastrow, Xt_lastrow], axis=1)

        Xt_lastrow = _coerce_col_str(Xt_lastrow)

        estimator = self.estimator_
        # 2D numpy array with col index = (fh, var) and 1 row
        y_pred = estimator.predict(Xt_lastrow)
        y_pred = y_pred.reshape((len(fh_idx), len(y_cols)))

        y_pred = pd.DataFrame(y_pred, columns=y_cols, index=fh_idx)

        if isinstance(y_pred.index, pd.MultiIndex):
            y_pred = y_pred.sort_index()

        return y_pred

    def _fit_concurrent(self, y, X=None, fh=None):
        """Fit to training data."""
        from sktime.transformations.series.impute import Imputer
        from sktime.transformations.series.lag import Lag

        impute_method = self.impute_method

        # lagger_y_to_X_ will lag y to obtain the sklearn X
        lags = self._lags
        lagger_y_to_X = Lag(lags=lags, index_out="extend")
        if impute_method is not None:
            lagger_y_to_X = lagger_y_to_X * Imputer(method=impute_method)
        self.lagger_y_to_X_ = lagger_y_to_X

        fh_rel = fh.to_relative(self.cutoff)
        y_lags = list(fh_rel)

        Xt = lagger_y_to_X.fit_transform(y)

        self.estimators_ = []

        for lag in y_lags:

            lag_plus = Lag(lag, index_out="extend")
            Xtt = lag_plus.fit_transform(Xt)
            Xtt_notna = Xtt.notnull().all(axis=1)
            Xtt_notna_idx = Xtt_notna.index[Xtt_notna].intersection(y.index)

            yt = y.loc[Xtt_notna_idx]
            Xtt = Xtt.loc[Xtt_notna_idx]

            # we now check whether the set of full lags is empty
            # if yes, we set a flag, since we cannot fit the reducer
            # instead, later, we return a dummy prediction
            if len(Xtt_notna_idx) == 0:
                self.estimators_.append(y.mean())
            else:
                if X is not None:
                    Xtt = pd.concat([X.loc[Xtt_notna_idx], Xtt], axis=1)

                Xtt = _coerce_col_str(Xtt)
                yt = _coerce_col_str(yt)

                estimator = clone(self.estimator)
                estimator.fit(Xtt, yt)
                self.estimators_.append(estimator)

        return self

    def _predict_concurrent(self, X=None, fh=None):
        """Fit to training data."""
        from sktime.transformations.series.lag import Lag

        if X is not None and self._X is not None:
            X_pool = X.combine_first(self._X)
        elif X is None and self._X is not None:
            X_pool = self._X
        else:
            X_pool = X

        fh_idx = self._get_expected_pred_idx(fh=fh)
        y_cols = self._y.columns

        lagger_y_to_X = self.lagger_y_to_X_

        fh_rel = fh.to_relative(self.cutoff)
        fh_abs = fh.to_absolute(self.cutoff)
        y_lags = list(fh_rel)
        y_abs = list(fh_abs)

        Xt = lagger_y_to_X.transform(self._y)
        y_pred_list = []

        for i in range(len(y_lags)):

            lag = y_lags[i]
            predict_idx = y_abs[i]

            lag_plus = Lag(lag, index_out="extend")
            Xtt = lag_plus.fit_transform(Xt)
            Xtt_predrow = slice_at_ix(Xtt, predict_idx)
            if X_pool is not None:
                Xtt_predrow = pd.concat(
                    [slice_at_ix(X_pool, predict_idx), Xtt_predrow], axis=1
                )

            Xtt_predrow = _coerce_col_str(Xtt_predrow)

            estimator = self.estimators_[i]

            # if = no training indices in _fit, fill in y training mean
            if isinstance(estimator, pd.Series):
                y_pred_i = pd.DataFrame(index=[0], columns=y_cols)
                y_pred_i.iloc[0] = estimator
            # otherwise proceed as per direct reduction algorithm
            else:
                y_pred_i = estimator.predict(Xtt_predrow)
            # 2D numpy array with col index = (var) and 1 row
            y_pred_list.append(y_pred_i)

        y_pred = np.concatenate(y_pred_list)
        y_pred = pd.DataFrame(y_pred, columns=y_cols, index=fh_idx)

        if isinstance(y_pred.index, pd.MultiIndex):
            y_pred = y_pred.sort_index()

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.linear_model import LinearRegression

        est = LinearRegression()
        params1 = {
            "estimator": est,
            "window_length": 3,
            "X_treatment": "shifted",
            "pooling": "global",  # all internal mtypes are tested across scenarios
        }
        params2 = {
            "estimator": est,
            "window_length": 3,
            "X_treatment": "concurrent",
            "pooling": "global",
        }
        return [params1, params2]
