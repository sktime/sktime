import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ..utils.validation import validate_forecasting_horizon
from ..utils.validation import validate_y_forecasting

__all__ = ["BaseForecaster", "BaseSingleSeriesForecaster", "BaseUpdateableForecaster"]
__author__ = ['Markus Löning']


class BaseForecaster(BaseEstimator):
    """Base class for forecasters.

    Parameters
    ----------
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """
    _estimator_type = "forecaster"

    def __init__(self, check_input=True):
        self.check_input = check_input
        self._y_idx = None
        self._is_fitted = False

    def fit(self, y, X=None):
        """Fit forecaster.

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
        if self.check_input:
            validate_y_forecasting(y)
            if X is not None:
                X = check_array(X)

        # keep index for later forecasting where passed forecasting horizon will be relative to y's index
        self._y_idx = self._get_y_index(y)

        # make interface compatible with estimators that only take y
        kwargs = {} if X is None else {'X': X}

        # Fit estimator.
        self._fit(y, **kwargs)
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None):
        """Predict using fitted estimator.

        Parameters
        ----------
        fh : array-like, optional (default=[1])
            The forecasting horizon with the steps ahead to to predict.
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

        # set default 1-step ahead forecast horizon
        fh = [1] if fh is None else fh

        if self.check_input:
            fh = validate_forecasting_horizon(fh)
            if X is not None:
                X = check_array(X)

        # make interface compatible with estimators that only take y
        kwargs = {} if X is None else {'X': X}

        # estimator specific implementation of fit method
        return self._predict(fh=fh, **kwargs)

    def score(self, y, fh=None, X=None, sample_weight=None):
        """Returns the mean squared error on the given forecast horizon.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        fh : array-like, optional (default=[1])
            The forecasting horizon with the steps ahead to to predict.
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
        # only check y here, X and fh will be checked in predict method
        if self.check_input:
            validate_y_forecasting(y)

        # predict y_pred
        # pass exogenous variable to predict only if passed as some forecasters may not accept X in predict
        kwargs = {} if X is None else {'X': X}
        y_pred = self.predict(fh=fh, **kwargs)

        # unnest y_true
        y_true = y.iloc[0]

        # check if passed true time series coincides with forecast horizon of predicted values
        if not y_true.index.equals(y_pred.index):
            raise ValueError(f"Index of passed time series ``y`` does not match index of predicted time series")

        return mean_squared_error(y_true, y_pred, sample_weight=sample_weight)

    @staticmethod
    def _get_y_index(y):
        """Helper function to get (time) index of y used in fitting for later comparison
        with forecast horizon"""
        y = y.iloc[0]
        index = y.index if hasattr(y, 'index') else pd.RangeIndex(len(y))
        return index


class BaseUpdateableForecaster(BaseForecaster):
    # TODO should that be a mixin class instead?
    """Base class for forecasters with update functionality.

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
        """Update forecasts using new data via Kalman smoothing/filtering of
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
            if X is not None:
                X = check_array(X)
            validate_y_forecasting(y)
            self._check_y_update(y)

        self._update(y, X=X)
        self._is_updated = True
        return self

    def _check_y_update(self, y):
        """Helper function to check the ``y`` passed to update the estimator"""
        y = y.iloc[0]
        y_idx = y.index if hasattr(y, 'index') else pd.RangeIndex(len(y))
        if not isinstance(y_idx, type(self._y_idx)):
            raise ValueError('Passed y does not match the initial y used for fitting')


class BaseSingleSeriesForecaster(BaseForecaster):
    """Classical forecaster which implements predict method for single-series/univariate fitted/updated classical
    forecasting techniques without exogenous variables (X).
    """

    def _predict(self, fh=None):
        """Internal predict

        Parameters
        ----------
        fh : array-like, optional (default=[1])
            The forecasting horizon with the steps ahead to to predict.

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """

        # Convert step-ahead prediction horizon into zero-based index
        fh_idx = fh - np.min(fh)

        # Predict fitted model with start and end points relative to start of train series
        fh = len(self._y_idx) - 1 + fh
        start = fh[0]
        end = fh[-1]
        y_pred = self._fitted_estimator.predict(start=start, end=end)

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        return y_pred.iloc[fh_idx]
