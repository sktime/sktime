#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License
"""Adapter for statsmodels ARIMA and SARIMAX models."""

__author__ = ["AryanDhanuka10", "fkiraly"]
__all__ = ["_StatsmodelsArimaAdapter"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError:
    SARIMAX = None


class _StatsmodelsArimaAdapter(BaseForecaster):
    """Adapter for statsmodels ARIMA and SARIMAX models.

    Parameters
    ----------
    order : tuple of int, default=(1, 0, 0)
        The (p, d, q) order of the model for AR, differencing, MA.
    seasonal_order : tuple of int, default=(0, 0, 0, 0)
        The (P, D, Q, s) seasonal order for SARIMAX.
    """

    _tags = {
        "python_dependencies": ["statsmodels"],
        "ignores-exogeneous-X": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
        "requires-fh-in-fit": False,
    }

    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit ARIMA/SARIMAX model using statsmodels."""
        if SARIMAX is None:
            raise ModuleNotFoundError(
                "statsmodels is required."
                " Please install it with `pip install statsmodels`."
            )

        # Align exogenous data with y
        if X is not None:
            X = self._check_X_index(y, X)

        self._model = SARIMAX(
            endog=y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self._fitted_model = self._model.fit(disp=False)
        return self

    def _predict(self, fh, X=None):
        """Make point predictions for given forecasting horizon."""
        n_periods = len(fh)

        # Align exogenous data if provided
        if X is not None:
            X = self._check_X_index(None, X)

        forecast_res = self._fitted_model.get_forecast(steps=n_periods, exog=X)
        preds = forecast_res.predicted_mean

        # Ensure output is a pandas Series with same index type as y
        return pd.Series(preds.values, index=fh.to_absolute(self.cutoff).to_pandas())

    def _predict_interval(self, fh, X=None, coverage=0.9):
        """Compute prediction intervals for the forecast."""
        n_periods = len(fh)

        if X is not None:
            X = self._check_X_index(None, X)

        forecast_res = self._fitted_model.get_forecast(steps=n_periods, exog=X)
        conf_int = forecast_res.conf_int(alpha=1 - coverage)

        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]

        idx = fh.to_absolute(self.cutoff).to_pandas()
        return {
            "lower": pd.Series(lower.values, index=idx),
            "upper": pd.Series(upper.values, index=idx),
        }

    def _update(self, y, X=None, update_params=True):
        """Update the model with new data without full refit."""
        if X is not None:
            X = self._check_X_index(y, X)

        self._fitted_model = self._fitted_model.append(y, exog=X, refit=update_params)
        return self

    def get_fitted_params(self):
        """Return fitted ARIMA parameters."""
        return self._fitted_model.params.to_dict()

    # --- Utility methods ---
    def _check_X_index(self, y, X):
        """Ensure X has same index as y (if provided)."""
        if y is not None and not X.index.equals(y.index):
            X = X.copy()
            X.index = y.index
        return X
