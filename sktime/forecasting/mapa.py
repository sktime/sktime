"""MAPA Forecaster implementation."""

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.series import check_series


class MAPAForecaster(BaseForecaster):
    """MAPA Forecaster class."""

    def __init__(
        self,
        aggregation_levels=None,
        base_forecaster=None,
        agg_method="mean",
        decompose_type="multiplicative",
        forecast_combine="mean",
        imputation_method="ffill",
        sp=6,
        weights=None,
        parallel=False,
        n_jobs=-1,
        conf_interval=0.95,
    ):
        self.aggregation_levels = (
            aggregation_levels if aggregation_levels else [1, 2, 4]
        )
        self.base_forecaster = (
            base_forecaster
            if base_forecaster is not None
            else ExponentialSmoothing(trend="add", seasonal="add", sp=sp)
        )
        self.agg_method = agg_method
        self.decompose_type = decompose_type
        self.forecast_combine = forecast_combine
        self.imputation_method = imputation_method
        self.sp = sp
        self.weights = weights
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.conf_interval = conf_interval

        self.forecasters = {}
        self.trend = {}
        self.seasonal = {}
        self.residuals = {}
        self.frequency = None
        self._fh = None
        self._transformation_offset = None

        super().__init__()

    def _handle_missing_data(self, y):
        if self.imputation_method == "ffill":
            y = y.ffill()
        elif self.imputation_method == "bfill":
            y = y.bfill()
        elif self.imputation_method == "interpolate":
            y = y.interpolate()
        else:
            raise ValueError(f"Unsupported imputation method: {self.imputation_method}")
        return y

    def _ensure_positive_values(self, y):
        if self.decompose_type == "multiplicative" and (y <= 0).any():
            min_positive_value = y[y > 0].min()
            offset = min_positive_value / 2
            y += offset
            self._transformation_offset = offset
            print(
                f"Applied an offset of {offset} to ensure positive values\
                 for multiplicative decomposition."
            )
        return y

    def _ensure_datetime_index(self, y):
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index, errors="coerce")
            if y.index.hasnans:
                raise ValueError(
                    "Could not convert index to pd.DatetimeIndex. \
                    Please provide a datetime-compatible index."
                )
        if y.index.freq is None:
            y = y.asfreq(pd.infer_freq(y.index))
        return y

    def _aggregate(self, y, level):
        if level == 1:
            return y
        return y.resample(f"{level}D").agg(self.agg_method).asfreq(y.index.freq)

    def _decompose(self, y):
        from sktime.utils.dependencies._dependencies import _check_soft_dependencies

        try:
            _check_soft_dependencies("statsmodels", severity="warning")
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            return

        y = self._ensure_positive_values(y)
        if y.isna().any():
            y = y.ffill().bfill()

        if len(y) < 2 * self.sp:
            return y, np.zeros_like(y), np.zeros_like(y)

        decomposition = seasonal_decompose(y, model=self.decompose_type, period=self.sp)
        return decomposition.trend, decomposition.seasonal, decomposition.resid

    def _combine_forecasts(self, forecasts):
        if self.forecast_combine == "mean":
            return np.mean(forecasts, axis=0)
        elif self.forecast_combine == "median":
            return np.median(forecasts, axis=0)
        elif self.forecast_combine == "weighted":
            weights = (
                self.weights
                if self.weights
                else [1 / level for level in self.aggregation_levels]
            )
            return np.average(forecasts, axis=0, weights=weights)
        else:
            raise ValueError("Unsupported forecast combination method.")

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        y = check_series(y)

        y = self._handle_missing_data(y)
        y = self._ensure_datetime_index(y)
        self.frequency = y.index.freq

        for level in self.aggregation_levels:
            y_agg = self._aggregate(y, level)
            trend, seasonal, residuals = self._decompose(y_agg)

            self.trend[level] = trend.ffill().bfill()
            self.seasonal[level] = seasonal.ffill().bfill()
            self.residuals[level] = residuals.ffill().bfill()

            forecaster = type(self.base_forecaster)(**self.base_forecaster.get_params())
            forecaster.fit(self.residuals[level])
            self.forecasters[level] = forecaster

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon."""
        fh = check_fh(fh)
        forecasts = []

        for level in self.aggregation_levels:
            forecaster = self.forecasters[level]
            residual_pred = forecaster.predict(fh)

            trend = self.trend[level].reindex(residual_pred.index, method="ffill")
            seasonal = self.seasonal[level].reindex(residual_pred.index, method="ffill")

            combined_pred = trend + seasonal + residual_pred

            if self._transformation_offset:
                combined_pred -= self._transformation_offset

            forecasts.append(combined_pred)

        final_forecast = self._combine_forecasts(forecasts)
        return pd.Series(final_forecast, index=fh.to_absolute(self.cutoff).to_pandas())

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data."""
        y = check_series(y)
        y = self._handle_missing_data(y)
        y = self._ensure_datetime_index(y)

        for level in self.aggregation_levels:
            y_agg = self._aggregate(y, level)
            trend, seasonal, residuals = self._decompose(y_agg)

            self.trend[level] = pd.concat([self.trend[level], trend]).ffill().bfill()
            self.seasonal[level] = (
                pd.concat([self.seasonal[level], seasonal]).ffill().bfill()
            )
            self.residuals[level] = (
                pd.concat([self.residuals[level], residuals]).ffill().bfill()
            )

            if update_params:
                self.forecasters[level].update(residuals)

        return self

    def _custom_exp_smoothing_update(self, forecaster, new_residuals):
        forecaster._y = pd.concat([forecaster._y, new_residuals])
        forecaster._is_fitted = False
        forecaster.fit(forecaster._y)
        return forecaster

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "aggregation_levels": [1, 2, 3],
            "base_forecaster": ExponentialSmoothing(trend="add", seasonal="add", sp=6),
            "imputation_method": "ffill",
            "decompose_type": "multiplicative",
            "forecast_combine": "mean",
        }
        params2 = {
            "aggregation_levels": [1, 4, 6],
            "base_forecaster": ExponentialSmoothing(trend="add", seasonal="mul", sp=6),
            "imputation_method": "interpolate",
            "decompose_type": "additive",
            "forecast_combine": "median",
        }
        return [params1, params2]
