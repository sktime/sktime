from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from ..highlevel import ForecastingTask


class BaseForecaster(BaseEstimator):
    """Base class for forecasters, for identification.
    """
    _estimator_type = "forecaster"

    def __init__(self, check_input=True):
        self.check_input = check_input
        self._is_fitted = False
        self._is_updated = False

    def fit(self, task, data):
        if self.check_input:
            self._check_fit_data(data)
            self._check_task(task)
        self.task = task

        target = data[self.task.target].iloc[0]
        self._target_idx = target.index if hasattr(target, 'index') else pd.RangeIndex(len(target))

        data = self._transform_data(data)
        self._fit(data)  # forecaster specific implementation
        self._is_fitted = True
        return self

    def update(self, data):
        check_is_fitted(self, '_is_fitted')
        if self.check_input:
            self._check_update_data(data)

        data = self._transform_data(data)
        self._update(data)  # forecaster specific implementation
        return self

    def predict(self):
        check_is_fitted(self, '_is_fitted')
        return self._predict()  # forecaster specific implementation

    def _transform_data(self, data):
        """Helper function to transform nested data with series/arrays in cells into pd.Series with primitives in cells
        """
        return pd.Series(*data[self.task.target].tolist())

    @staticmethod
    def _check_fit_data(data):
        # TODO input checks for forecasting
        #  regularly spaced, date/time index or numeric
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data must be supplied as a pandas DataFrame, but found {type(data)}')
        if not data.shape[0] == 1:
            raise ValueError(f'Data must be from a single instance (row), but found {data.shape[0]} rows')

    def _check_update_data(self, data):
        target = data[self.task.target].iloc[0]
        updated_target_idx = target.index if hasattr(target, 'index') else pd.RangeIndex(len(target))
        is_longer = updated_target_idx.min() == self._target_idx.max()
        is_same_type = isinstance(updated_target_idx, type(self._target_idx))
        if not (is_longer and is_same_type):
            raise ValueError('Data passed to `update` does not match the data passed to `fit`')

    @staticmethod
    def _check_task(task):
        # TODO input checks for forecasting
        if not isinstance(task, ForecastingTask):
            raise ValueError(f'Task must be ForecastingTask, but found f{type(task)}')


class ClassicalForecaster(BaseForecaster):
    """Classical forecaster which implements predict method for fitted/updated classical forecasting techniques.
    """

    def _predict(self):
        # Convert step-ahead prediction horizon into zero-based index
        pred_horizon = self.task.pred_horizon
        pred_horizon_idx = pred_horizon - np.min(pred_horizon)

        if self._is_updated:
            # Predict updated (pre-initialised) model with start and end values relative to end of train series
            start = self.task.pred_horizon[0]
            end = self.task.pred_horizon[-1]
            pred = self.updated_model.predict(start=start, end=end)

        else:
            # Predict fitted model with start and end points relative to start of train series
            pred_horizon = pred_horizon + len(self._target_idx) - 1
            start = pred_horizon[0]
            end = pred_horizon[-1]
            pred = self.fitted_model.predict(start=start, end=end)

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        return pred[pred_horizon_idx]
