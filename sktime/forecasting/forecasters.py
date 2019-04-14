from .base import BaseForecaster

import pandas as pd

# SARIMAX is better maintained and offers same functionality as ARIMA class
# https://github.com/statsmodels/statsmodels/issues/3884
from statsmodels.tsa.statespace.sarimax import SARIMAX


class ARMAForecaster(BaseForecaster):
    def __init__(self, order=None, check_input=True):
        self._check_order(order, 2)
        self.order = order
        self._arima_order = self.order + (0,)  # make into equivalent order for more general ARIMA model

        super(ARMAForecaster, self).__init__(check_input=check_input)

    def _fit(self, data):
        self.model = SARIMAX(data, order=self._arima_order)
        self.fitted_model = self.model.fit()

    def _predict(self, start, end):
        return self.fitted_model.predict(start=start, end=end)

    def _update(self, data):
        # Update data and model.
        self.model = SARIMAX(self.data, order=self._arima_order)

        # Filter given fitted parameters.
        self.updated_model = self.model.filter(self.fitted_model.params)


class ARIMAForecaster(BaseForecaster):
    def __init__(self, order=None, trend='n', enforce_stationarity=True, enforce_invertibility=True, check_input=True):
        self._check_order(order, 3)
        self.order = order
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        super(ARIMAForecaster, self).__init__(check_input=check_input)

    def _fit(self, data):
        self.model = SARIMAX(data,
                             order=self.order,
                             trend=self.trend,
                             enforce_stationarity=self.enforce_stationarity,
                             enforce_invertibility=self.enforce_invertibility)
        self.fitted_model = self.model.fit()

    def _predict(self, start, end):
        return self.fitted_model.predict(start=start, end=end)

    def _update(self, data):
        # Update data and model.
        self.model = SARIMAX(data, order=self.order)

        # Filter given fitted parameters.
        self.updated_model = self.model.filter(self.fitted_model.params)


