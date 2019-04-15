from .base import BaseForecaster
import numpy as np

# SARIMAX is better maintained and offers same functionality as ARIMA class
# https://github.com/statsmodels/statsmodels/issues/3884
from statsmodels.tsa.statespace.sarimax import SARIMAX

__all__ = ["ARIMAForecaster"]


class ARIMAForecaster(BaseForecaster):
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

    def _update(self, data):
        # TODO for updating see https://github.com/statsmodels/statsmodels/issues/2788

        # Update model.
        self.model = SARIMAX(data,
                             order=self.order,
                             trend=self.trend,
                             enforce_stationarity=self.enforce_stationarity,
                             enforce_invertibility=self.enforce_invertibility)
        self.model.initialize_known(self.fitted_model.predicted_state[:, -2],
                                    self.fitted_model.predicted_state_cov[:, :, -2])

        # Filter given fitted parameters.
        self.updated_model = self.model.smooth(self.fitted_model.params)
        self._is_updated = True


