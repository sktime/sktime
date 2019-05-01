from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd

from ..utils.validation import check_forecasting_horizon
from ..utils.validation import check_y_forecasting

__all__ = ["BaseForecaster", "_SingleSeriesForecaster"]
__author__ = ['Markus Löning']


class BaseForecaster(BaseEstimator):
    """Base class for forecasters, for identification.
    """
    _estimator_type = "forecaster"

    def __init__(self, check_input=True):
        self.check_input = check_input

    def score(self, y, fh=None, X=None, sample_weight=None):
        if self.check_input:
            check_y_forecasting(y)
        y = y.iloc[0]

        # pass exogenous variable to predict only if passed as some forecasters may not accept X in predict
        kwargs = {} if X is None else {'exog': X}

        return mean_squared_error(y, self.predict(fh=fh, **kwargs), sample_weight=sample_weight)

    @staticmethod
    def _get_index(y):
        index = y.index if hasattr(y, 'index') else pd.RangeIndex(stop=len(y))
        return index


class _SingleSeriesForecaster(BaseForecaster):
    """Classical forecaster which implements predict method for single-series/univariate fitted/updated classical
    forecasting techniques without exogenous variables.
    """

    def fit(self, y, X=None):
        if self.check_input:
            check_y_forecasting(y)

        # unnest series
        y = y.iloc[0]

        # keep index for later forecasting where passed forecasting horizon will be relative to y's index
        self._y_idx = self._get_index(y)

        return self._fit(y)

    def predict(self, fh=None):
        # Convert step-ahead prediction horizon into zero-based index
        check_is_fitted(self, '_fitted_estimator')
        fh = [1] if fh is None else fh

        if self.check_input:
            fh = check_forecasting_horizon(fh)

        fh_idx = fh - np.min(fh)

        if hasattr(self, '_updated_estimator'):
            # Predict updated (pre-initialised) model with start and end values relative to end of train series
            start = fh[0]
            end = fh[-1]
            y_pred = self._updated_estimator.predict(start=start, end=end)

        else:
            # Predict fitted model with start and end points relative to start of train series
            fh = len(self._y_idx) - 1 + fh
            start = fh[0]
            end = fh[-1]
            y_pred = self._fitted_estimator.predict(start=start, end=end)

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        return y_pred.iloc[fh_idx]


