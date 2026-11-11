# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Unified base class for zero-shot pretrained time series foundation models."""

__all__ = ["BasePretrainedForecaster"]
__author__ = ["Lakshya-2440"]

import numpy as np
import pandas as pd
from scipy.stats import norm

from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.utils.dependencies import _check_soft_dependencies


class BasePretrainedForecaster(BaseForecaster):
    """Unified base class for zero-shot time series foundation models.

    Provides a standardized interface for pretrained time series foundation models
    (such as Chronos, Moment, UniTS, TabPFN, FlowState, T0) with context caching,
    device management, zero-shot forecasting, and quantile extraction.

    Parameters
    ----------
    model_name : str or None, default=None
        Name or path of the pretrained foundation model checkpoint (e.g. HuggingFace
        hub ID or local path).
    device : str or None, default=None
        Device for model computation (e.g., "cpu", "cuda", "mps"). If None,
        automatically falls back to "cuda" if CUDA is available, then "mps",
        otherwise "cpu".
    context_length : int or None, default=None
        Maximum context window length to retain from historical time series. If the
        input series exceeds this length, only the most recent ``context_length``
        time points are cached and fed to the foundation model. If None, the entire
        provided context is preserved.
    """

    _tags = {
        "authors": ["Lakshya-2440"],
        "maintainers": ["Lakshya-2440"],
        "category": ["deep_learning", "foundation_model"],
        "capability:zero_shot": True,
        "requires-fh-in-fit": False,
        "capability:predict_proba": True,
        "capability:pred_int": True,
        "scitype:y": "both",
        "capability:multivariate": True,
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "capability:insample": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        context_length: int | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length
        self._context_series = None
        self._context_X = None
        self._device = None

        super().__init__()

    @property
    def device_(self) -> str:
        """Resolved device for tensor execution with CPU/CUDA fallback."""
        if self.device is not None:
            return self.device
        if self._device is not None:
            return self._device
        resolved = self._resolve_device()
        self._device = resolved
        return resolved

    def _resolve_device(self) -> str:
        """Resolve available hardware accelerator with safe fallback."""
        if _check_soft_dependencies("torch", severity="none"):
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    @property
    def context_series(self):
        """Cached historical context series for zero-shot forecasting."""
        return getattr(self, "_context_series", None)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster by caching historical context series.

        Foundation models require no task-specific parameter optimization during fit;
        instead, fit caches the historical context series, sets cutoff, and marks
        the estimator as fitted.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Historical context series.
        X : pd.Series or pd.DataFrame, optional (default=None)
            Exogenous time series.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        context = y
        if self.context_length is not None and len(context) > self.context_length:
            context = context.iloc[-self.context_length :]

        self._context_series = context
        self._context_X = X
        if (
            (not hasattr(self, "_cutoff") or self._cutoff is None)
            and hasattr(y, "index")
            and len(y.index) > 0
        ):
            self._set_cutoff_from_y(y)
        self._is_fitted = True
        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon given cached context.

        Parameters
        ----------
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon encoding the time stamps to forecast at.
        X : pd.Series or pd.DataFrame, optional (default=None)
            Exogenous time series for the forecast.

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
            Point predictions at horizon fh.
        """
        if self._context_series is None:
            raise ValueError(
                f"{self.__class__.__name__} has no context series cached. "
                "Call fit(y) first or invoke predict_zero_shot(y, fh)."
            )

        context = self._context_series
        if self.context_length is not None and len(context) > self.context_length:
            context = context.iloc[-self.context_length :]

        return self._predict_zero_shot(context=context, fh=fh, X=X)

    def _predict_zero_shot(self, context, fh=None, X=None):
        """Generate zero-shot forecasts from context and forecasting horizon.

        Subclasses should override this method to interface with foundation model
        inference pipelines. If not overridden, provides default mock autoregressive
        behavior (last observed value carry-forward).

        Parameters
        ----------
        context : pd.Series or pd.DataFrame
            Historical context series.
        fh : ForecastingHorizon, optional (default=None)
            The forecasting horizon with time points to predict.
        X : pd.Series or pd.DataFrame, optional (default=None)
            Exogenous time series.

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
            Point predictions matching fh index and context columns.
        """
        if fh is None:
            fh = getattr(self, "fh", None)
        if fh is None:
            raise ValueError("Forecasting horizon `fh` must be provided.")

        cutoff = getattr(self, "cutoff", None)
        if cutoff is None and hasattr(context, "index") and len(context.index) > 0:
            cutoff = context.index[-1]

        if not isinstance(fh, ForecastingHorizon):
            fh = self._check_fh(fh)

        fh_abs = fh.to_absolute(cutoff)
        pred_index = fh_abs.to_pandas()
        n_steps = len(pred_index)

        if isinstance(context, pd.DataFrame):
            last_vals = context.iloc[-1].values
            data = np.tile(last_vals, (n_steps, 1))
            return pd.DataFrame(data, index=pred_index, columns=context.columns)
        elif isinstance(context, pd.Series):
            last_val = context.iloc[-1]
            return pd.Series(last_val, index=pred_index, name=context.name)
        else:
            return pd.Series(context[-1], index=pred_index)

    def _predict_quantiles(self, fh=None, X=None, alpha=None, **kwargs):
        """Compute quantile forecasts from model distribution or context dispersion.

        Subclasses can override this method or ``_predict_quantiles_zero_shot`` to
        extract empirical quantiles from the model's predictive head. The base
        implementation uses Gaussian distribution quantiles around point forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon.
        X : pd.Series or pd.DataFrame, optional (default=None)
            Exogenous series.
        alpha : list of float, optional (default=[0.05, 0.95])
            Quantile levels in (0, 1).
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        quantiles : pd.DataFrame
            Quantile forecasts with 2-level MultiIndex columns (var_name, alpha).
        """
        # Handle positional call where alpha is passed as first argument
        if (
            isinstance(fh, (list, tuple, np.ndarray, float))
            and alpha is None
            and not isinstance(fh, ForecastingHorizon)
        ):
            alpha = fh
            fh = kwargs.get("fh", None)
            X = kwargs.get("X", X)

        if alpha is None:
            alpha = [0.05, 0.95]
        alpha = sorted(alpha)

        if hasattr(self, "_predict_quantiles_zero_shot"):
            context = self._context_series
            if self.context_length is not None and len(context) > self.context_length:
                context = context.iloc[-self.context_length :]
            return self._predict_quantiles_zero_shot(
                context=context, fh=fh, X=X, alpha=alpha
            )

        y_pred = self._predict(fh=fh, X=X)
        varnames = self._get_varnames()
        columns = self._get_columns("predict_quantiles", alpha=alpha)

        cutoff = getattr(self, "cutoff", None)
        if cutoff is None and self._context_series is not None:
            cutoff = self._context_series.index[-1]
        fh_abs = fh.to_absolute(cutoff)
        pred_index = fh_abs.to_pandas()

        y_pred_df = (
            y_pred.to_frame(name=varnames[0])
            if isinstance(y_pred, pd.Series)
            else y_pred
        )

        data = {}
        for col in varnames:
            if (
                isinstance(self._context_series, pd.DataFrame)
                and col in self._context_series
            ):
                series = self._context_series[col]
            elif isinstance(self._context_series, pd.Series):
                series = self._context_series
            else:
                series = pd.Series(self._context_series)

            diffs = series.diff().dropna()
            if len(diffs) > 1 and float(diffs.std()) > 1e-8:
                std = float(diffs.std())
            elif len(series) > 1 and float(series.std()) > 1e-8:
                std = float(series.std())
            else:
                std = 1.0

            for a in alpha:
                z = float(norm.ppf(a))
                data[(col, a)] = y_pred_df[col].values + z * std

        quantiles_df = pd.DataFrame(data, index=pred_index, columns=columns)
        return quantiles_df

    def predict_zero_shot(self, y, fh=None, X=None):
        """Perform zero-shot forecasting without prior fit call.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Historical context series.
        fh : int, list, np.ndarray, pd.Index, or ForecastingHorizon, default=None
            Forecasting horizon.
        X : pd.Series or pd.DataFrame, optional (default=None)
            Exogenous series.

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
            Zero-shot forecast.
        """
        self.fit(y=y, X=X, fh=fh)
        return self.predict(fh=fh, X=X)

    def predict_quantiles_zero_shot(self, y, fh=None, X=None, alpha=None):
        """Perform zero-shot quantile forecasting without prior fit call.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Historical context series.
        fh : int, list, np.ndarray, pd.Index, or ForecastingHorizon, default=None
            Forecasting horizon.
        X : pd.Series or pd.DataFrame, optional (default=None)
            Exogenous series.
        alpha : list of float, optional (default=[0.05, 0.95])
            Quantile levels in (0, 1).

        Returns
        -------
        quantiles : pd.DataFrame
            Quantile forecasts.
        """
        self.fit(y=y, X=X, fh=fh)
        return self.predict_quantiles(fh=fh, X=X, alpha=alpha)

    def predict(self, fh=None, X=None, y=None):
        """Forecast time series at future horizon.

        If ``y`` is provided, it is established as historical context zero-shot
        without requiring an explicit prior ``fit`` call.

        Parameters
        ----------
        fh : int, list, np.ndarray, pd.Index, or ForecastingHorizon, default=None
            The forecasting horizon encoding the time stamps to forecast at.
        X : pd.Series or pd.DataFrame, optional (default=None)
            Exogenous time series to use in prediction.
        y : pd.Series or pd.DataFrame, optional (default=None)
            Optional historical context series for direct zero-shot prediction.

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
            Point forecasts at fh.
        """
        if y is not None:
            self.fit(y=y, X=X, fh=fh)
        return super().predict(fh=fh, X=X)

    def predict_quantiles(self, fh=None, X=None, alpha=None, y=None):
        """Compute quantile forecasts.

        If ``y`` is provided, it is established as historical context zero-shot
        without requiring an explicit prior ``fit`` call.

        Parameters
        ----------
        fh : int, list, np.ndarray, pd.Index, or ForecastingHorizon, default=None
            Forecasting horizon.
        X : pd.Series or pd.DataFrame, optional (default=None)
            Exogenous series.
        alpha : list of float, optional (default=[0.05, 0.95])
            Quantile levels.
        y : pd.Series or pd.DataFrame, optional (default=None)
            Optional historical context series for direct zero-shot prediction.

        Returns
        -------
        quantiles : pd.DataFrame
            Quantile forecasts.
        """
        if y is not None:
            self.fit(y=y, X=X, fh=fh)
        return super().predict_quantiles(fh=fh, X=X, alpha=alpha)
