from .base import BaseSingleSeriesForecaster
from .base import BaseForecaster
from .base import BaseUpdateableForecaster

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
# SARIMAX is better maintained and offers same functionality as ARIMA class
# https://github.com/statsmodels/statsmodels/issues/3884


__all__ = ["ARIMAForecaster", "ExpSmoothingForecaster", "DummyForecaster", "EnsembleForecaster"]
__author__ = ['Markus Löning']


class ARIMAForecaster(BaseUpdateableForecaster):
    def __init__(self, order, seasonal_order=None, trend='n', enforce_stationarity=True, enforce_invertibility=True, check_input=True,
                 maxiter=1000, method='lbfgs', disp=5):

        self._check_order(order, 3)
        self.order = order

        if seasonal_order is not None:
            self._check_order(seasonal_order, 4)
            self.seasonal_order = seasonal_order
        else:
            self.seasonal_order = (0, 0, 0, 0)

        self.trend = trend
        self.method = method
        self.disp = disp
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter
        super(ARIMAForecaster, self).__init__(check_input=check_input)

    def _fit(self, y, X=None):
        """Fit forecaster.

        Parameters
        ----------
        y
        X

        Returns
        -------

        """
        # unnest series
        y = y.iloc[0]

        # fit estimator
        self._estimator = SARIMAX(y,
                                  exog=X,
                                  order=self.order,
                                  seasonal_order=self.seasonal_order,
                                  trend=self.trend,
                                  enforce_stationarity=self.enforce_stationarity,
                                  enforce_invertibility=self.enforce_invertibility)
        self._fitted_estimator = self._estimator.fit(maxiter=self.maxiter, method=self.method, disp=self.disp)
        return self

    def _update(self, y, X=None):
        """Update forecasts using Kalman smoothing on passed updated data and forecasts based on previously fitted
        parameters"""
        # TODO for updating see https://github.com/statsmodels/statsmodels/issues/2788 and
        #  https://github.com/statsmodels/statsmodels/issues/3318

        # unnest series
        y = y.iloc[0]

        # Update estimator.
        estimator = SARIMAX(y,
                            exog=X,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            trend=self.trend,
                            enforce_stationarity=self.enforce_stationarity,
                            enforce_invertibility=self.enforce_invertibility)
        estimator.initialize_known(self._fitted_estimator.predicted_state[:, -1],
                                   self._fitted_estimator.predicted_state_cov[:, :, -1])

        # Filter given fitted parameters.
        self._updated_estimator = estimator.smooth(self._fitted_estimator.params)
        return self

    def _predict(self, fh=None, X=None):
        """Make predictions

        Parameters
        ----------
        fh
        X

        Returns
        -------

        """

        fh_idx = fh - np.min(fh)

        if self._is_updated:
            # Predict updated (pre-initialised) model with start and end values relative to end of train series
            start = fh[0]
            end = fh[-1]
            y_pred = self._updated_estimator.predict(start=start, end=end, exog=X)

        else:
            # Predict fitted model with start and end points relative to start of train series
            fh = len(self._y_idx) - 1 + fh
            start = fh[0]
            end = fh[-1]
            y_pred = self._fitted_estimator.predict(start=start, end=end, exog=X)

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        return y_pred.iloc[fh_idx]

    @staticmethod
    def _check_order(order, n):
        if not (isinstance(order, tuple) and (len(order) == n)):
            raise ValueError(f'Order must be a tuple of length f{n}')

        if not all(np.issubdtype(type(k), np.integer) for k in order):
            raise ValueError(f'All values in order must be integers')


class ExpSmoothingForecaster(BaseSingleSeriesForecaster):
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
        super(ExpSmoothingForecaster, self).__init__(check_input=check_input)

    def _fit(self, y, X=None):
        """Fit forecaster

        Parameters
        ----------
        y
        X

        Returns
        -------

        """
        # unnest series
        y = y.iloc[0]

        self.estimator = ExponentialSmoothing(y, trend=self.trend, damped=self.damped, seasonal=self.seasonal,
                                              seasonal_periods=self.seasonal_periods)
        self._fitted_estimator = self.estimator.fit(smoothing_level=self.smoothing_level, optimized=self.optimized,
                                                    smoothing_slope=self.smoothing_slope,
                                                    smoothing_seasonal=self.smooting_seasonal, damping_slope=self.damping_slope,
                                                    use_boxcox=self.use_boxcox, remove_bias=self.remove_bias,
                                                    use_basinhopping=self.use_basinhopping)
        return self


class DummyForecaster(BaseSingleSeriesForecaster):
    def __init__(self, strategy='mean', check_input=True):

        allowed_strategies = ('mean', 'last', 'seasonal_last', 'linear')
        if strategy not in allowed_strategies:
            raise ValueError(f'Unknown strategy: {strategy}, expected one of {allowed_strategies}')

        self.strategy = strategy
        super(DummyForecaster, self).__init__(check_input=check_input)

    def _fit(self, y, X=None):
        """Fit forecaster

        Parameters
        ----------
        y
        X

        Returns
        -------

        """
        # unnest series
        y = y.iloc[0]

        if self.strategy == 'mean':
            self.estimator = ExponentialSmoothing(y)
            self._fitted_estimator = self.estimator.fit(smoothing_level=0)

        if self.strategy == 'last':
            self.estimator = ExponentialSmoothing(y)
            self._fitted_estimator = self.estimator.fit(smoothing_level=1)

        if self.strategy == 'seasonal_last':
            # TODO how to pass seasonal frequency for ARIMA estimator
            # if not hasattr(data, 'index'):
            #     raise ValueError('Cannot determine seasonal frequency from index of passed data')
            # s = data.index.freq
            # self.estimator = SARIMAX(data, seasonal_order=(0, 1, 0, s), trend='n')
            # self._fitted_estimator = self.estimator.fit()
            raise NotImplementedError()

        if self.strategy == 'linear':
            self.estimator = SARIMAX(y, order=(0, 0, 0), trend='t')
            self._fitted_estimator = self.estimator.fit()

        return self


class EnsembleForecaster(BaseForecaster):
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
        super(EnsembleForecaster, self).__init__(check_input=check_input)

    def _fit(self, y, X=None):
        """Fit forecaster"""
        for _, estimator in self.estimators:
            estimator.set_params(**{"check_input": False})
            fitted_estimator = estimator.fit(y, X=X)
            self.fitted_estimators_.append(fitted_estimator)
        return self

    def _predict(self, fh=None, X=None):
        """Make predictions

        Parameters
        ----------
        fh
        X

        Returns
        -------

        """
        fh_idx = fh - np.min(fh)

        # Iterate over estimators
        y_preds = np.zeros((len(self.fitted_estimators_), len(fh)))
        indexes = []
        for i, estimator in enumerate(self.fitted_estimators_):
            # TODO pass X to estimators where the predict method accepts X
            y_pred = estimator.predict(fh=fh)[fh_idx]
            y_preds[i, :] = y_pred.values
            indexes.append(y_pred.index)

        # Check predicted horizons
        if not all(index.equals(indexes[0]) for index in indexes):
            raise ValueError('Predicted horizons from estimators do not match')

        # Average predictions
        avg_preds = np.average(y_preds, axis=0, weights=self.weights)

        # Return average predictions with index
        index = indexes[0]
        name = y_preds[0].name if hasattr(y_preds[0], 'name') else None
        return pd.Series(avg_preds, index=index, name=name)


