from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array
import numpy as np
import pandas as pd

from ..utils.validation import check_forecasting_horizon
from ..utils.validation import check_y_forecasting

__all__ = ["BaseForecaster", "BaseSingleSeriesForecaster", "BaseUpdateableForecaster"]
__author__ = ['Markus Löning']


class BaseForecaster(BaseEstimator):
    """Base class for forecasters, for identification.
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
        y
        X

        Returns
        -------

        """
        if self.check_input:
            check_y_forecasting(y)
            if X is not None:
                X = check_array(X)

        # keep index for later forecasting where passed forecasting horizon will be relative to y's index
        self._y_idx = self._get_y_index(y)

        self._fit(y, X=X)
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None):
        """Make predictions.

        Parameters
        ----------
        fh : array-like
            Forecasting horizon
        X : pandas DataFrame
            Exogeneous data

        Returns
        -------
        Predictions
        """
        check_is_fitted(self, '_is_fitted')

        # set default 1-step ahead forecast horizon
        fh = [1] if fh is None else fh

        if self.check_input:
            fh = check_forecasting_horizon(fh)
            if X is not None:
                X = check_array(X)

        # make interface compatible with estimators that only take y and not X
        kwargs = {} if X is None else {'X': X}
        return self._predict(fh=fh, **kwargs)

    def score(self, y, fh=None, X=None, sample_weight=None):
        """Default score method

        Parameters
        ----------
        y
        fh
        X
        sample_weight

        Returns
        -------

        """
        if self.check_input:
            check_y_forecasting(y)

        # unnest y
        y = y.iloc[0]

        # pass exogenous variable to predict only if passed as some forecasters may not accept X in predict
        kwargs = {} if X is None else {'X': X}

        return mean_squared_error(y, self.predict(fh=fh, **kwargs), sample_weight=sample_weight)

    @staticmethod
    def _get_y_index(y):
        """Helper function to keep track of time index of y used in fitting"""
        y = y.iloc[0]
        index = y.index if hasattr(y, 'index') else pd.RangeIndex(stop=len(y))
        return index


class BaseUpdateableForecaster(BaseForecaster):
    # TODO should that be a mixin class instead?

    def __init__(self, check_input=True):
        super(BaseUpdateableForecaster, self).__init__(check_input=check_input)
        self._is_updated = False

    def update(self, y, X=None):
        """Update forecasts using new data.

        Parameters
        ----------
        y
        X

        Returns
        -------

        """
        check_is_fitted(self, '_is_fitted')
        if self.check_input:
            if X is not None:
                X = check_array(X)
            check_y_forecasting(y)
            self._check_y_update(y)

        self._update(y, X=X)
        self._is_updated = True
        return self

    def _check_y_update(self, y):
        """Helper function to check y passed to update estimator"""
        y = y.iloc[0]
        y_idx = y.index if hasattr(y, 'index') else pd.RangeIndex(len(y))
        if not isinstance(y_idx, type(self._y_idx)):
            raise ValueError('Passed y does not match the initial y used for fitting')


class BaseSingleSeriesForecaster(BaseForecaster):
    """Classical forecaster which implements predict method for single-series/univariate fitted/updated classical
    forecasting techniques without exogenous variables (X).
    """

    def _predict(self, fh=None):

        # Convert step-ahead prediction horizon into zero-based index
        fh_idx = fh - np.min(fh)

        # Predict fitted model with start and end points relative to start of train series
        fh = len(self._y_idx) - 1 + fh
        start = fh[0]
        end = fh[-1]
        y_pred = self._fitted_estimator.predict(start=start, end=end)

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        return y_pred.iloc[fh_idx]


