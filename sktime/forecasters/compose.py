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
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """

    # TODO: experimental, major functionality not implemented (input checks, params interface, exogenous variables)

    def __init__(self, estimators=None, weights=None, check_input=True):
        # TODO add input checks
        self.estimators = estimators
        self.weights = weights
        self.fitted_estimators_ = []
        super(EnsembleForecaster, self).__init__(check_input=check_input)

    def fit(self, y, fh=None, X=None):
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

        for _, estimator in self.estimators:
            # TODO implement set/get params interface
            # estimator.set_params(**{"check_input": False})
            fitted_estimator = estimator.fit(y, fh=fh, X=X)
            self.fitted_estimators_.append(fitted_estimator)
        return self

    def predict(self, fh=None, X=None):
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

    def __init__(self, estimator, window_length=None, recursive=False, check_input=True):
        self.estimator = estimator
        self.window_length = window_length
        self.recursive = recursive
        super(ReducedTimeSeriesRegressionForecaster, self).__init__(check_input=check_input)

    def fit(self, y, fh=None, X=None):
        """Fit forecaster.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        fh : array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])
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
        if fh is None and not self.recursive:
            raise ValueError(f"If dynamic is set to False, forecasting horizon (fh) has to be specified in fit, "
                             f"as one estimator is fit for each step of the forecasting horizon")

        if fh is not None:
            fh = validate_fh(fh)

        if X is not None:
            # TODO concatenate exogeneous variables X to rolled window matrix X below
            raise NotImplementedError()

        # Unnest series
        yt = self._prepare_y(y)

        # Transform using rolling window
        X, Y = self.transform(yt, fh)

        # Fitting
        if self.recursive:
            # Fit single estimator for one-step ahead forecast
            # which is then used iteratively when predicting
            y = Y.ravel()  # convert into one-dimensional array
            estimator = clone(self.estimator)
            estimator.fit(X, y)
            self.estimators_ = estimator

        else:
            # Fit one estimator for each step-ahead forecast
            self.estimators_ = []
            len_fh = len(fh)

            # Iterate over estimators/forecast horizon
            for i in range(len_fh):
                estimator = clone(self.estimator)
                y = pd.Series(Y[:, i])
                estimator.fit(X, y)
                self.estimators_.append(estimator)

        # Save the last window-length number of observations for predicting
        self.window_length_ = self.rw.get_window_length()
        self._last_window = yt.iloc[-self.window_length_:]
        self._is_fitted = True
        return self

    def transform(self, y, fh):
        """Helper function to transform data using rolling window approach"""

        # Set up window roller
        # For dynamic prediction, models are only trained on one-step ahead forecast
        fh = np.array([1]) if self.recursive else fh
        self.rw = RollingWindowSplit(window_length=self.window_length, fh=fh)

        # get numeric time index
        time_index = y.index.values
        if not np.issubdtype(time_index.dtype, np.dtype(int).type):
            raise NotImplementedError("Non-numeric time indices are not supported yet")

        # Transform target series into tabular format using rolling window splits
        xs = []
        ys = []
        for feature_window, target_window in self.rw.split(time_index):
            xi = y[feature_window]
            yi = y[target_window]
            xs.append(xi)
            ys.append(yi)

        # Construct nested pandas DataFrame for X
        X = pd.DataFrame(pd.Series([np.asarray(xi) for xi in xs]))
        Y = np.array([np.asarray(yi) for yi in ys])
        return X, Y

    def predict(self, fh=None, X=None):

        if X is not None:
            # TODO handle exog data
            raise NotImplementedError()

        # get forecasting horizon
        fh = validate_fh(fh)
        len_fh = len(fh)

        # use last window as test data for prediction
        x_test = pd.DataFrame(pd.Series([self._last_window]))
        y_pred = np.zeros(len(fh))

        # prediction can be either dynamic making only one-step ahead forecasts using previous forecasts or static using
        # only the last window and using one fitted estimator for each step ahead forecast
        if self.recursive:
            # Roll last window using previous one-step ahead forecasts
            for i in range(len_fh):
                y_pred[i] = self.estimators_.predict(x_test)

                # append prediction to last window and roll window
                x_test = np.append(x_test.iloc[0, 0].values, y_pred[i])[-self.window_length_:]

                # put data into required nested format
                x_test = pd.DataFrame(pd.Series([pd.Series(x_test)]))

        else:
            # Iterate over estimators/forecast horizon
            # Any fh is ignored if specified
            for i, estimator in enumerate(self.estimators_):
                y_pred[i] = estimator.predict(x_test)

        # Add name and forecast index
        index = self._last_window.index[-1] + fh
        name = self._last_window.name

        return pd.Series(y_pred, name=name, index=index)


