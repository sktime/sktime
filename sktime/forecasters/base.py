import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted

from sktime.utils.validation import validate_fh
from sktime.utils.data_container import get_time_index, tabularise


__all__ = ["BaseForecaster", "BaseSingleSeriesForecaster", "BaseUpdateableForecaster"]
__author__ = ['Markus Löning']


class BaseForecaster(BaseEstimator):
    """
    Base class for forecasters.

    Parameters
    ----------
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    _estimator_type = "forecaster"

    def __init__(self, check_input=True):
        self.check_input = check_input
        self._time_index = None
        self._is_fitted = False

    def fit(self, y, fh=None, X=None):
        """
        Fit forecaster.

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
        if self.check_input:
            self._validate_X_y(y, X)

        # validate forecasting horizon
        fh = validate_fh(fh)

        # Keep index for predicting where forecasters horizon will be relative to y seen in fit
        self._time_index = get_time_index(y)

        # Make interface compatible with estimators that only take y and no X
        kwargs = {} if X is None else {'X': X}

        # Internal fit.
        self._fit(y, fh=fh, **kwargs)
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None):
        """
        Predict using fitted estimator.

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
        check_is_fitted(self, '_is_fitted')

        if self.check_input:
            self._validate_X(X)

        # validate forecasters horizon
        fh = validate_fh(fh)

        # make interface compatible with estimators that only take y
        kwargs = {} if X is None else {'X': X}

        # estimator specific implementation of fit method
        return self._predict(fh=fh, **kwargs)

    def score(self, y, fh=None, X=None, sample_weight=None):
        """
        Returns the root mean squared error on the given forecast horizon.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        fh : array-like, optional (default=[1])
            The forecasters horizon with the steps ahead to to predict.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must also be provided
            exogenous features for making predictions.
        sample_weight : array-like, shape = [n_observations], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean squared error of self.predict(fh=fh, X=X) with respect to y.
        """
        # only check y here, X and fh will be checked during predict
        if self.check_input:
            self._validate_y(y)

        # Predict y_pred
        # pass exogenous variable to predict only if given, as some forecasters may not accept X in predict
        kwargs = {} if X is None else {'X': X}
        y_pred = self.predict(fh=fh, **kwargs)

        # Unnest y_true
        y_true = y.iloc[0]

        # Check if passed true time series coincides with forecast horizon of predicted values
        if not y_true.index.equals(y_pred.index):
            raise ValueError(f"Index of passed time series `y` does not match index of predicted time series; "
                             f"make sure the forecasters horizon `fh` matches the time index of `y`")

        return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

    def _validate_X_y(self, y, X=None):
        """Helper function to check input data for forecasters

        Parameters
        ----------
        y : pandas.Series
            Time series to forecast.
        X : pandas.DataFrame
            Dataframe with exogenous data
        """
        # TODO add more input checks for consistency of X and y
        self._validate_y(y)
        self._validate_X(X)

    @staticmethod
    def _validate_y(y):
        """
        Helper function to check input data for forecasters

        Parameters
        ----------
        y : pandas.Series
            Time series to forecast.
        """
        # Check if pandas series
        if not isinstance(y, pd.Series):
            raise ValueError(f'``y`` must be a pandas Series, but found: {type(y)}')

        # Check if single row
        if not y.shape[0] == 1:
            raise ValueError(f'``y`` must consist of a pandas Series with a single row, '
                             f'but found: {y.shape[0]} rows')

        # Check if contained time series is either pandas series or numpy array
        s = y.iloc[0]
        if not isinstance(s, (np.ndarray, pd.Series)):
            raise ValueError(f'``y`` must contain a pandas Series or numpy array, but found: {type(s)}.')

    @staticmethod
    def _validate_X(X):
        """
        Helper function to check input data for forecasters

        Parameters
        ----------
        X : pandas.DataFrame
            Dataframe with exogenous data
        """
        if X is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"`X` must a pandas DataFrame, but found: {type(X)}")
            if X.shape[0] > 1:
                raise ValueError(f"`X` must consist of a single row, but found: {X.shape[0]} rows")

            # Check if index is the same for all columns.

            # Get index from first row, can be either pd.Series or np.array.
            first_index = X.iloc[0, 0].index if hasattr(X.iloc[0, 0], 'index') else pd.RangeIndex(X.iloc[0, 0].shape[0])

            # Series must contain at least 2 observations, otherwise should be primitive.
            if len(first_index) < 1:
                raise ValueError(f'Time series must contain at least 2 observations, but found: '
                                 f'{len(first_index)} observations in column: {X.columns[0]}')

            # Compare with remaining columns
            for c, col in enumerate(X.columns):
                index = X.iloc[0, c].index if hasattr(X.iloc[0, c], 'index') else pd.RangeIndex(X.iloc[0, 0].shape[0])
                if not np.array_equal(first_index, index):
                    raise ValueError(f'Found time series with unequal index in column {col}. '
                                     f'Input time-series must have the same index.')

    @staticmethod
    def _prepare_X(X):
        """Helper function to transform nested pandas DataFrame X into 2d numpy array as required by `statsmodels`
        estimators.

        Parameters
        ----------
        X : pandas.DataFrame, shape=[1, n_variables]
            Nested dataframe with series of shape [n_obs,] in cells

        Returns
        -------
        Xt : ndarray, shape=[n_obs, n_variables]
        """
        if X is None:
            return X

        if X.shape[1] > 1:
            Xl = X.iloc[0, :].tolist()
            Xt = np.column_stack(Xl)
        else:
            Xt = tabularise(X).values.T

        return Xt

    @staticmethod
    def _prepare_y(y):
        """Helper function to prepare y as required by `statsmodels` estimators.

        Parameters
        ----------
        y : pandas.Series, shape = [1,]
            Nested series with series of shape [n_obs,] in first cell

        Returns
        -------
        yt : pandas Series, shape=[n_obs,]
        """
        return y.iloc[0]


class BaseUpdateableForecaster(BaseForecaster):
    # TODO should that be a mixin class instead?
    """
    Base class for forecasters with update functionality.

    Parameters
    ----------
    check_input : bool, optional
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
        Default is True.
    """

    def __init__(self, check_input=True):
        super(BaseUpdateableForecaster, self).__init__(check_input=check_input)
        self._is_updated = False

    def update(self, y, X=None):
        """
        Update forecasts using new data via Kalman smoothing/filtering of
        forecasts obtained from previously fitted forecaster.

        Parameters
        ----------
        y : pandas.Series
            Updated time series which to use for updating the previously fitted forecaster.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must also be provided
            exogenous features for making predictions.

        Returns
        -------
        self : An instance of self
        """
        check_is_fitted(self, '_is_fitted')
        if self.check_input:
            self._validate_X_y(y, X)
            self._validate_y_update(y)

        self._update(y, X=X)
        self._is_updated = True
        return self

    def _validate_y_update(self, y):
        """
        Helper function to check the ``y`` passed to update the estimator
        """
        # TODO add input checks for X when updating
        # TODO add additional input checks for update data, i.e. that update data is newer than data seen in fit
        y = y.iloc[0]
        y_idx = y.index if hasattr(y, 'index') else pd.RangeIndex(len(y))
        if not isinstance(y_idx, type(self._time_index)):
            raise ValueError('Passed y does not match the initial y used for fitting')


class BaseSingleSeriesForecaster(BaseForecaster):
    """
    Classical forecaster which implements predict method for single-series/univariate fitted/updated classical
    forecasters techniques without exogenous variables (X).
    """

    def _predict(self, fh=None):
        """
        Internal predict.

        Parameters
        ----------
        fh : array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """

        # Convert step-ahead prediction horizon into zero-based index
        fh_idx = fh - np.min(fh)

        # Predict fitted model with start and end points relative to start of train series
        fh = len(self._time_index) - 1 + fh
        start = fh[0]
        end = fh[-1]
        y_pred = self._fitted_estimator.predict(start=start, end=end)

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        return y_pred.iloc[fh_idx]
