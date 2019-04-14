from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from ..highlevel import ForecastingTask


class BaseForecaster(BaseEstimator):
    """
    Base class for forecasters, for identification.
    """
    _estimator_type = "forecaster"

    def __init__(self, check_input=True):
        self.check_input = check_input

    def fit(self, task, data):
        if self.check_input:
            self._check_input_data(data)
            self._check_task(task)

        self.task = task
        self.data = data

        data = self._transform_data(data)
        self._fit(data)  # forecaster specific implementation
        self._is_fitted = True
        return self

    def update(self, data):
        check_is_fitted(self, '_is_fitted')
        if self.check_input:
            self._check_input_data(data)

        self.data = self._concat_data(data)

        data = self._transform_data(data)
        self._update(data)  # forecaster specific implementation
        return self

    def predict(self):
        check_is_fitted(self, '_is_fitted')

        pred_horizon = self.task.pred_horizon
        pred_horizon_idx = np.asarray(pred_horizon) - 1  # zero-indexing
        m = len(self.data.iloc[0])
        start = m + pred_horizon[0]
        end = m + pred_horizon[-1]
        preds = self._predict(start, end)  # forecaster specific implementation
        return preds.iloc[pred_horizon_idx]

    def _concat_data(self, data):
        concat = pd.concat([self._transform_data(self.data),
                            self._transform_data(data)], axis=0)
        return pd.Series([concat])

    @staticmethod
    def _transform_data(data):
        """Helper function to transform nested data with series/arrays in cells into pd.Series with primitives in cells
        """
        return pd.Series(*data.tolist())

    @staticmethod
    def _check_order(order, n):
        if not (isinstance(order, tuple) and (len(order) == n)):
            raise ValueError()
        if not all(np.issubdtype(type(k), np.integer) for k in order):
            raise ValueError()

    @staticmethod
    def _check_input_data(data):
        # TODO input checks for forecasting
        if not isinstance(data, pd.Series):
            raise ValueError()

    @staticmethod
    def _check_task(task):
        # TODO input checks for forecasting
        if not isinstance(task, ForecastingTask):
            raise ValueError()



