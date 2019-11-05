__author__ = "Markus Löning"
__all__ = ["EnsembleForecaster",
           "TransformedTargetForecaster",
           "ReducedTimeSeriesRegressionForecaster"]


import pandas as pd
import numpy as np
from sklearn.base import clone
from sktime.utils.validation.forecasting import check_integer_time_index
from sktime.utils.validation.forecasting import validate_fh
from sktime.forecasters.model_selection import RollingWindowSplit
from sktime.forecasters.base import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """
    Meta-estimator for ensembling forecasters.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples.
    weights : array-like, shape = [n_estimators], optional (default=None)
        Sequence of weights (float or int) to weight the occurrences of predicted values before averaging.
        Uses uniform weights if None.
    """

    # TODO: experimental, major functionality not implemented (input checks, params interface, exogenous variables)

    def __init__(self, estimators=None, weights=None):
        # TODO add input checks
        self.estimators = estimators
        self.weights = weights
        self.fitted_estimators_ = []
        super(EnsembleForecaster, self).__init__()

    def _fit(self, y, fh=None, X=None):
        """
        Internal fit.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.

        Returns
        -------
        self : returns an instance of self.
        """
        # Clear previously fitted estimators
        self.fitted_estimators_ = []

        for _, estimator in self.estimators:
            # TODO implement set/get params interface
            fitted_estimator = estimator.fit(y, fh=fh, X=X)
            self.fitted_estimators_.append(fitted_estimator)
        return self

    def _predict(self, fh=None, X=None):
        """
        Internal predict using fitted estimator.

        Parameters
        ----------
        fh : array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.

        Returns
        -------
        Predictions : pandas.Series, shape=(len(fh),)
            Returns series of predicted values.
        """
        # TODO pass X only to estimators where the predict method accepts X, currenlty X is ignored

        # Iterate over estimators
        y_preds = np.zeros((len(self.fitted_estimators_), len(fh)))
        indexes = []
        for i, estimator in enumerate(self.fitted_estimators_):
            y_pred = estimator.predict(fh=fh)
            y_preds[i, :] = y_pred
            indexes.append(y_pred.index)

        # Check if all predicted horizons are identical
        if not all(index.equals(indexes[0]) for index in indexes):
            raise ValueError('Predicted horizons from estimators do not match')

        # Average predictions over estimators
        avg_preds = np.average(y_preds, axis=0, weights=self.weights)

        # Return average predictions with index
        index = indexes[0]
        name = y_preds[0].name if hasattr(y_preds[0], 'name') else None
        return pd.Series(avg_preds, index=index, name=name)

    def get_params(self, deep=True):
        # TODO fix get and set params interface following sklearn double underscore convention
        raise NotImplementedError()

    def set_params(self, **params):
        raise NotImplementedError()


class TransformedTargetForecaster(BaseForecaster):
    """Meta-estimator for forecasting transformed time series."""

    # TODO add check inverse method after fitting transformer

    def __init__(self, forecaster, transformer):
        self.forecaster = forecaster
        self.transformer = transformer
        super(TransformedTargetForecaster, self).__init__()

    def transform(self, y):
        # transformers are designed to modify X which is 2-dimensional, we
        # need to modify y accordingly.
        y = pd.DataFrame(pd.Series([y]))

        self.transformer_ = clone(self.transformer)
        yt = self.transformer_.fit_transform(y)

        # restore 1d target
        yt = yt.iloc[0, 0]
        return yt

    def inverse_transform(self, y):
        # transformers are designed to modify nested X which is 2-dimensional, we
        # need to modify y accordingly.
        y = pd.DataFrame(pd.Series([y]))
        yit = self.transformer_.inverse_transform(y)

        # restore 1d target
        yit = yit.iloc[0, 0]
        return yit

    def _fit(self, y, fh=None, X=None):
        """Internal fit"""
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._input_shape = y.ndim

        # transform data
        yt = self.transform(y)

        # fit forecaster using transformed target data
        self.forecaster_ = clone(self.forecaster)
        self.forecaster_.fit(yt, fh=fh, X=X)
        return self

    def _predict(self, fh=None, X=None):
        """Predict"""
        y_pred = self.forecaster_.predict(fh=fh, X=X)

        # compute inverse transform
        y_pred_it = self.inverse_transform(y_pred)

        # return unnested format
        return y_pred_it


class ReducedTimeSeriesRegressionForecaster(BaseForecaster):
    """
    Meta-estimator for forecasting by reduction to time series regression.

    Strategy to reduce a forecasters problem to a time series regression
    problem using a rolling window approach

    Parameters
    ----------
    ts_regressor : a time series regressor
    window_length : int, optional (default=None)
        Window length of rolling window approach.
    recursive : bool, optional (default=False)
        - If True, estimator is fitted for one-step ahead forecasts and only one-step ahead forecasts are made using
        extending the last window of the training data with already made forecasts.
        - If False, one estimator is fitted for each step-ahead forecast and only the last window is used for making
        forecasts.
    """

    def __init__(self, ts_regressor, window_length=None, recursive=False):
        self.ts_regressor = ts_regressor
        self.window_length = window_length
        self.recursive = recursive
        self.rw = None
        self.estimators_ = []
        super(ReducedTimeSeriesRegressionForecaster, self).__init__()

    def _fit(self, y, fh=None, X=None):
        """Internal fit.

        self : returns an instance of self.
        """
        # input checks
        # this forecaster requires the forecasting horizon already in fit
        if fh is None and not self.recursive:
            raise ValueError(f"If recursive is set to False, the forecasting horizon (fh) "
                             f"must be specified in fit, as one estimator is fit for each "
                             f"step of the forecasting horizon")

        if X is not None:
            # TODO concatenate exogeneous variables X to rolled window matrix X below
            raise NotImplementedError()

        # reset list of fitted estimators
        self.estimators_ = []

        # Transform input time series using rolling window tabularisation
        Xs, Ys = self.transform(y, fh)

        # Fitting of recursive strategy: fit single estimator for one-step ahead forecast
        # which is then used iteratively in prediction
        if self.recursive:
            ys = pd.Series(Ys.ravel())  # convert into one-dimensional array
            estimator = clone(self.ts_regressor)
            estimator.fit(Xs, ys)
            self.estimators_.append(estimator)

        # Fitting of non-recursive strategy: fitting one estimator for each step-ahead forecast
        else:
            # Iterate over estimators/forecast horizon
            for i in range(len(fh)):
                estimator = clone(self.ts_regressor)
                ys = pd.Series(Ys[:, i])
                estimator.fit(Xs, ys)
                self.estimators_.append(estimator)

        # Store the last window-length number of observations for prediction
        self.window_length_ = self.rw.get_window_length()
        self._last_window = y.iloc[-self.window_length_:]
        return self

    def transform(self, y, fh=None):
        """Transform data using rolling window approach"""
        # check input
        if fh is not None:
            fh = validate_fh(fh)

        # Set up window roller
        # for recursive reduction strategy, the forecaster are fitted on
        # one-step ahead forecast and then recursively applied to make
        # forecasts for the whole forecasting horizon
        if self.recursive:
            rw_fh = np.array([1])
        else:
            rw_fh = fh
        self.rw = RollingWindowSplit(window_length=self.window_length, fh=rw_fh)

        # get integer time index
        time_index = check_integer_time_index(self._time_index)

        # Transform target series into tabular format using
        # rolling window tabularisation
        xs = []
        ys = []
        for feature_window, target_window in self.rw.split(time_index):
            xi = y[feature_window]
            yi = y[target_window]
            xs.append(xi)
            ys.append(yi)

        # Put into nested dataframe format for time series regression
        X = pd.DataFrame(pd.Series([np.asarray(xi) for xi in xs]))
        Y = np.array([np.asarray(yi) for yi in ys])
        return X, Y

    def _predict(self, fh=None, X=None):

        # check input
        if X is not None:
            raise NotImplementedError()

        # use last window as new input data for time series regressors to make forecasts
        y_last = pd.DataFrame(pd.Series([self._last_window]))

        # preallocate array for forecasted values
        len_fh = len(fh)
        y_pred = np.zeros(len_fh)

        # forecasts can be either recursive or non-recursive;
        # for recursive forecasts, we make one-step ahead forecasts, and when making multi-step forecasts,
        # we use previously forecasted values to forecast the next step ahead;
        # for non-recursive forecasts, we use the last window of the training series and
        # fit one estimator for each step of the forecasting horizon
        if self.recursive:
            estimator = self.estimators_[0]
            for i in range(len_fh):
                y_pred[i] = estimator.predict(y_last)

                # append prediction to last window and roll window
                y_last = np.append(y_last.iloc[0, 0].values, y_pred[i])[-self.window_length_:]
                # put data into required nested format
                y_last = pd.DataFrame(pd.Series([pd.Series(y_last)]))

        # Iterate over estimators/forecast horizon
        else:
            for i, estimator in enumerate(self.estimators_):
                y_pred[i] = estimator.predict(y_last)

        # Add name and forecasting time index to forecasted series
        index = self._last_window.index[-1] + fh
        name = self._last_window.name

        return pd.Series(y_pred, name=name, index=index)


