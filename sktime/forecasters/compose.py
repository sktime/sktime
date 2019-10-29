import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from sktime.utils.validation.forecasting import validate_fh
from sktime.forecasters.model_selection import RollingWindowSplit
from sktime.forecasters.base import BaseForecaster

__author__ = "Markus Löning"
__all__ = ["EnsembleForecaster",
           "TransformedTargetForecaster",
           "ReducedTimeSeriesRegressionForecaster"]


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble of forecasters.

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

    def fit(self, y, fh=1, X=None):
        """
        Internal fit.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must also be provided
            exogenous features for making predictions.

        Returns
        -------
        self : returns an instance of self.
        """
        # validate forecasting horizon
        fh = validate_fh(fh)

        # Clear previously fitted estimators
        self.fitted_estimators_ = []

        for _, estimator in self.estimators:
            # TODO implement set/get params interface
            fitted_estimator = estimator.fit(y, fh=fh, X=X)
            self.fitted_estimators_.append(fitted_estimator)
        return self

    def predict(self, fh=1, X=None):
        """
        Internal predict using fitted estimator.

        Parameters
        ----------
        fh : array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that if
            provided, the forecaster must also have been fitted on the exogenous
            features.

        Returns
        -------
        Predictions : pandas.Series, shape=(len(fh),)
            Returns series of predicted values.
        """
        # TODO pass X only to estimators where the predict method accepts X, currenlty X is ignored

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        fh = validate_fh(fh)
        fh_idx = fh - np.min(fh)

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
    """Meta-estimator to forecast on a transformed target."""

    # TODO add check inverse method after fitting transformer

    def __init__(self, forecaster, transformer):
        self.forecaster = forecaster
        self.transformer = transformer

    def _transform(self, y):
        # transformers are designed to modify X which is 2-dimensional, we
        # need to modify y accordingly.
        y = pd.DataFrame(y) if isinstance(y, pd.Series) else y

        self.transformer_ = clone(self.transformer)
        yt = self.transformer_.fit_transform(y)

        # restore 1d target
        yt = yt.iloc[:, 0]
        return yt

    def _inverse_transform(self, y):
        # transformers are designed to modify X which is 2-dimensional, we
        # need to modify y accordingly.
        y = pd.DataFrame(y) if isinstance(y, pd.Series) else y
        yit = self.transformer_.inverse_transform(y)

        # restore 1d target
        yit = yit.iloc[:, 0]
        return yit

    def fit(self, y, fh=None, X=None):
        """Fit"""
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._input_shape = y.ndim

        # transform data
        yt = self._transform(y)

        # fit forecaster using transformed target data
        self.forecaster_ = clone(self.forecaster)
        self.forecaster_.fit(yt, fh=fh, X=X)
        return self

    def predict(self, fh=None, X=None):
        """Predict"""
        check_is_fitted(self, "forecaster_")
        y_pred = self.forecaster_.predict(fh=fh, X=X)

        # return to nested format
        y_pred = pd.Series([y_pred])

        # compute inverse transform
        y_pred_it = self._inverse_transform(y_pred)

        # return unnested format
        return y_pred_it.iloc[0]


class ReducedTimeSeriesRegressionForecaster(BaseForecaster):
    """
    Forecasting to time series regression reduction strategy.

    Strategy to reduce a forecasters problem to a time series regression
    problem using a rolling window approach

    Parameters
    ----------
    estimator : an estimator
        Time series regressor.
    window_length : int, optional (default=None)
        Window length of rolling window approach.
    recursive : bool, optional (default=False)
        - If True, estimator is fitted for one-step ahead forecasts and only one-step ahead forecasts are made using
        extending the last window of the training data with already made forecasts.
        - If False, one estimator is fitted for each step-ahead forecast and only the last window is used for making
        forecasts.
    """

    def __init__(self, estimator, window_length=None, recursive=False):
        self.estimator = estimator
        self.window_length = window_length
        self.recursive = recursive
        self.rw = None
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

        # Unnest series
        yt = self._prepare_y(y)

        # Transform input time series using rolling window tabularisation
        X, Y = self.transform(yt, fh)

        # Fitting of recursive strategy: fit single estimator for one-step ahead forecast
        # which is then used iteratively in prediction
        if self.recursive:
            y = Y.ravel()  # convert into one-dimensional array
            estimator = clone(self.estimator)
            estimator.fit(X, y)
            self.estimators_ = estimator

        # Fitting of non-recursive strategy: fitting one estimator for each step-ahead forecast
        else:
            self.estimators_ = []
            len_fh = len(fh)

            # Iterate over estimators/forecast horizon
            for i in range(len_fh):
                estimator = clone(self.estimator)
                y = pd.Series(Y[:, i])
                estimator.fit(X, y)
                self.estimators_.append(estimator)

        # Store the last window-length number of observations for prediction
        self.window_length_ = self.rw.get_window_length()
        self._last_window = yt.iloc[-self.window_length_:]
        self._is_fitted = True
        return self

    def transform(self, y, fh=None):
        """Helper function to transform data using rolling window approach"""
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

        # get numeric time index
        time_index = self._time_index.values
        if not np.issubdtype(time_index.dtype, np.dtype(int).type):
            raise NotImplementedError("Non-integer time indices are not supported yet")

        # Transform target series into tabular format using rolling window tabularisation
        xs = []
        ys = []
        for feature_window, target_window in self.rw.split(time_index):
            xi = y[feature_window]
            yi = y[target_window]
            xs.append(xi)
            ys.append(yi)

        # Construct nested pandas DataFrame for X for time series regression
        X = pd.DataFrame(pd.Series([np.asarray(xi) for xi in xs]))
        Y = np.array([np.asarray(yi) for yi in ys])
        return X, Y

    def _predict(self, fh=None, X=None):

        # check input
        if X is not None:
            raise NotImplementedError()

        len_fh = len(fh)

        # use last window as new input data for time series regressors to make forecasts
        x_new = pd.DataFrame(pd.Series([self._last_window]))
        y_pred = np.zeros(len(fh))

        # prediction can be either recursive making only one-step ahead forecasts
        # using previous forecasts or static using only the last window and
        # using one fitted estimator for each step ahead forecast
        if self.recursive:
            # Roll last window using previous one-step ahead forecasts
            for i in range(len_fh):
                y_pred[i] = self.estimators_.predict(x_new)

                # append prediction to last window and roll window
                x_new = np.append(x_new.iloc[0, 0].values, y_pred[i])[-self.window_length_:]

                # put data into required nested format
                x_new = pd.DataFrame(pd.Series([pd.Series(x_new)]))

        else:
            # Iterate over estimators/forecast horizon
            # Any fh is ignored if specified
            for i, estimator in enumerate(self.estimators_):
                y_pred[i] = estimator.predict(x_new)

        # Add name and forecast index
        index = self._last_window.index[-1] + fh
        name = self._last_window.name

        return pd.Series(y_pred, name=name, index=index)


