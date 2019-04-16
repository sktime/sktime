from .base import BaseForecaster
from .base import ClassicalForecaster

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# SARIMAX is better maintained and offers same functionality as ARIMA class
# https://github.com/statsmodels/statsmodels/issues/3884
from statsmodels.tsa.statespace.sarimax import SARIMAX


__all__ = ["ARIMAForecaster", "ExponentialSmoothingForecaster", "ForecastingEnsemble"]


class ARIMAForecaster(ClassicalForecaster):
    def __init__(self, order=None, trend='n', enforce_stationarity=True, enforce_invertibility=True, check_input=True,
                 maxiter=1000):
        self._check_order(order, 3)
        self.order = order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter
        super(ARIMAForecaster, self).__init__(check_input=check_input)

    def _fit(self, data):
        self.model = SARIMAX(data,
                             order=self.order,
                             trend=self.trend,
                             enforce_stationarity=self.enforce_stationarity,
                             enforce_invertibility=self.enforce_invertibility)
        self.fitted_model = self.model.fit(maxiter=self.maxiter)

    def _update(self, data):
        # TODO for updating see https://github.com/statsmodels/statsmodels/issues/2788

        # Update model.
        model = SARIMAX(data,
                        order=self.order,
                        trend=self.trend,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility)
        model.initialize_known(self.fitted_model.predicted_state[:, -2],
                               self.fitted_model.predicted_state_cov[:, :, -2])

        # Filter given fitted parameters.
        self.updated_model = model.smooth(self.fitted_model.params)
        self._is_updated = True

    @staticmethod
    def _check_order(order, n):
        if not (isinstance(order, tuple) and (len(order) == n)):
            raise ValueError(f'Order must be a tuple of length f{n}')
        if not all(np.issubdtype(type(k), np.integer) for k in order):
            raise ValueError(f'All values in order must be integers')


class ExponentialSmoothingForecaster(ClassicalForecaster):
    """Holt-Winters exponential smoothing forecaster. Default setting use simple exponential smoothing
    without trend and seasonality components.

    Parameters
    ----------
    trend
    damped
    seasonal
    seasonal_periods
    smoothing_level
    smoothing_slope
    smoothing_seasonal
    damping_slope
    optimized
    use_boxcox
    remove_bias
    use_basinhopping
    check_input
    """

    def __init__(self, trend=None, damped=False, seasonal=None, seasonal_periods=None, smoothing_level=None,
                 smoothing_slope=None, smoothing_seasonal=None, damping_slope=None, optimized=True,
                 use_boxcox=False, remove_bias=False, use_basinhopping=False, check_input=True):

        # Model params
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

        # Fit params
        self.smoothing_level = smoothing_level
        self.optimized = optimized
        self.smoothing_slope = smoothing_slope
        self.smooting_seasonal = smoothing_seasonal
        self.damping_slope = damping_slope
        self.use_boxcox = use_boxcox
        self.remove_bias = remove_bias
        self.use_basinhopping = use_basinhopping
        super(ExponentialSmoothingForecaster, self).__init__(check_input=check_input)

    def _fit(self, data):
        self.model = ExponentialSmoothing(data, trend=self.trend, damped=self.damped, seasonal=self.seasonal,
                                          seasonal_periods=self.seasonal_periods)
        self.fitted_model = self.model.fit(smoothing_level=self.smoothing_level, optimized=self.optimized,
                                           smoothing_slope=self.smoothing_slope,
                                           smoothing_seasonal=self.smooting_seasonal, damping_slope=self.damping_slope,
                                           use_boxcox=self.use_boxcox, remove_bias=self.remove_bias,
                                           use_basinhopping=self.use_basinhopping)
        return self

    def _update(self, data):
        # TODO
        raise NotImplementedError()


class ForecastingEnsemble(BaseForecaster):
    """Ensemble of forecasters.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples.
    weights :
    estimator_params
    estimator_fit_params
    check_input
    """

    def __init__(self, estimators=None, weights=None, check_input=True):
        self.estimators = estimators
        self.weights = weights
        self.estimators_ = []
        self.fitted_estimators_ = []
        super(ForecastingEnsemble, self).__init__(check_input=check_input)

    def fit(self, task, data):
        if self.check_input:
            self._check_fit_data(data)
            self._check_task(task)
        self.task = task

        target = data[self.task.target].iloc[0]
        self._target_idx = target.index if hasattr(target, 'index') else pd.RangeIndex(len(target))

        # Fit models
        for _, estimator in self.estimators_:
            fitted_estimator = estimator.fit(self.task, data)
            self.fitted_estimators_.append(fitted_estimator)

        self._is_fitted = True
        return self

    def update(self, data):
        raise NotImplementedError()

    def _predict(self):
        pred_horizon = self.task.pred_horizon
        pred_horizon_idx = pred_horizon - np.min(pred_horizon)

        # Predict fitted model with start and end points relative to start of train series
        pred_horizon = pred_horizon + len(self._target_idx) - 1
        start = pred_horizon[0]
        end = pred_horizon[-1]

        # Iterate over estimators
        n_estimators = len(self.estimators_)
        len_pred_horizon = len(pred_horizon)
        preds = np.zeros((n_estimators, len_pred_horizon))
        indexes = []
        for i, estimator in enumerate(self.fitted_estimators_):
            pred = estimator.predict(start=start, end=end)[pred_horizon_idx]
            preds[i, :] = pred.values
            indexes.append(pred.index)

        # Average predictions
        avg_preds = np.average(preds, axis=0, weights=self.weights)

        # Get index
        if not all(index == indexes[0] for index in indexes):
            raise ValueError()

        # Return average predictions with index
        return pd.Series(avg_preds, index=indexes[0], name=self.task.target)

