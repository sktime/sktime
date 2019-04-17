from ..highlevel import _BaseForecastingStrategy
from ..highlevel import _ClassicalSingleSeriesForecastingStrategy

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# SARIMAX is better maintained and offers same functionality as ARIMA class
# https://github.com/statsmodels/statsmodels/issues/3884
from statsmodels.tsa.statespace.sarimax import SARIMAX


__all__ = ["ARIMAForecaster", "ExponentialSmoothingForecaster", "ForecastingEnsemble"]


class ARIMAForecaster(_ClassicalSingleSeriesForecastingStrategy):
    def __init__(self, order=None, trend='n', enforce_stationarity=True, enforce_invertibility=True, check_input=True,
                 maxiter=1000):
        self._check_order(order, 3)
        self.order = order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter
        super(ARIMAForecaster, self).__init__(check_input=check_input)

    def _fit_estimator(self, data):
        self.model = SARIMAX(data,
                             order=self.order,
                             trend=self.trend,
                             enforce_stationarity=self.enforce_stationarity,
                             enforce_invertibility=self.enforce_invertibility)
        self._fitted_model = self.model.fit(maxiter=self.maxiter)
        return self

    def _update(self, data):
        # TODO for updating see https://github.com/statsmodels/statsmodels/issues/2788

        # Update model.
        model = SARIMAX(data,
                        order=self.order,
                        trend=self.trend,
                        enforce_stationarity=self.enforce_stationarity,
                        enforce_invertibility=self.enforce_invertibility)
        model.initialize_known(self._fitted_model.predicted_state[:, -2],
                               self._fitted_model.predicted_state_cov[:, :, -2])

        # Filter given fitted parameters.
        self._updated_model = model.smooth(self._fitted_model.params)
        self._is_updated = True

    @staticmethod
    def _check_order(order, n):
        if not (isinstance(order, tuple) and (len(order) == n)):
            raise ValueError(f'Order must be a tuple of length f{n}')
        if not all(np.issubdtype(type(k), np.integer) for k in order):
            raise ValueError(f'All values in order must be integers')


class ExponentialSmoothingForecaster(_ClassicalSingleSeriesForecastingStrategy):
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

    def _fit_estimator(self, data):
        self.model = ExponentialSmoothing(data, trend=self.trend, damped=self.damped, seasonal=self.seasonal,
                                          seasonal_periods=self.seasonal_periods)
        self._fitted_model = self.model.fit(smoothing_level=self.smoothing_level, optimized=self.optimized,
                                           smoothing_slope=self.smoothing_slope,
                                           smoothing_seasonal=self.smooting_seasonal, damping_slope=self.damping_slope,
                                           use_boxcox=self.use_boxcox, remove_bias=self.remove_bias,
                                           use_basinhopping=self.use_basinhopping)
        return self

    def _update(self, data):
        # TODO
        raise NotImplementedError()


class DummyForecaster(_ClassicalSingleSeriesForecastingStrategy):
    def __init__(self, strategy='mean', check_input=True):

        allowed_strategies = ('mean', 'last', 'seasonal_last', 'linear')
        if strategy not in allowed_strategies:
            raise ValueError(f'Unknown strategy: {strategy}, expected one of {allowed_strategies}')

        self.strategy = strategy
        super(DummyForecaster, self).__init__(check_input=check_input)

    def _fit_estimator(self, data):
        if self.strategy == 'mean':
            self.model = ExponentialSmoothing(data)
            self._fitted_model = self.model.fit(smoothing_level=0)
        if self.strategy == 'last':
            self.model = ExponentialSmoothing(data)
            self._fitted_model = self.model.fit(smoothing_level=1)
        if self.strategy == 'seasonal_last':
            # TODO how to pass seasonal frequency for ARIMA model
            # if not hasattr(data, 'index'):
            #     raise ValueError('Cannot determine seasonal frequency from index of passed data')
            # s = data.index.freq
            # self.model = SARIMAX(data, seasonal_order=(0, 1, 0, s), trend='n')
            # self._fitted_model = self.model.fit()
            raise NotImplementedError()
        if self.strategy == 'linear':
            self.model = SARIMAX(data, order=(0, 0, 0), trend='t')
            self._fitted_model = self.model.fit()
        return self

    def _update(self, data):
        # TODO
        raise NotImplementedError()


class ForecastingEnsemble(_BaseForecastingStrategy):
    """Ensemble of forecasters.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples.
    weights :
    check_input :
    """

    def __init__(self, estimators=None, weights=None, check_input=True):
        self.estimators = estimators
        self.weights = weights
        self.fitted_estimators_ = []
        super(ForecastingEnsemble, self).__init__(check_input=check_input)

    def _fit_strategy(self, data):
        # Fit models
        for _, estimator in self.estimators:
            fitted_estimator = estimator.fit(self._task, data)
            self.fitted_estimators_.append(fitted_estimator)
        return self

    def _update(self, data):
        raise NotImplementedError()

    def _predict(self):
        pred_horizon = self._task.pred_horizon
        pred_horizon_idx = pred_horizon - np.min(pred_horizon)

        # Iterate over estimators
        preds = np.zeros((len(self.fitted_estimators_), len(pred_horizon)))
        indexes = []
        for i, estimator in enumerate(self.fitted_estimators_):
            pred = estimator.predict()[pred_horizon_idx]
            preds[i, :] = pred.values
            indexes.append(pred.index)

        # Check predicted horizons
        if not all(index.equals(indexes[0]) for index in indexes):
            raise ValueError('Predicted horizons from estimators do not match')

        # Average predictions
        avg_preds = np.average(preds, axis=0, weights=self.weights)

        # Return average predictions with index
        index = indexes[0]
        return pd.Series(avg_preds, index=index, name=self._task.target)
