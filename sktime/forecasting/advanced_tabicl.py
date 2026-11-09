# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Advanced TabICL forecaster for sktime.

This module implements ``AdvancedTabICLForecaster``, a univariate forecaster that
uses reduction: it transforms a time series into tabular sliding-window features,
then delegates learning to ``tabicl.TabICLRegressor``.

The design intentionally separates:

- forecasting mechanics (horizon validation, recursive/direct rollout), and
- model mechanics (model class lookup, parameter preparation, KV-cache handling),

to keep the implementation extensible for future tabular foundation-model backends.
"""

__author__ = ["sktime developers"]
__all__ = ["AdvancedTabICLForecaster"]

from copy import deepcopy
from collections.abc import Mapping

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _check_soft_dependencies


class AdvancedTabICLForecaster(BaseForecaster):
    """TabICL-powered reduction forecaster for univariate time series.

    The estimator applies a reduction strategy:

    1. build sliding-window tabular features from ``y``;
    2. fit ``TabICLRegressor`` on this derived supervised dataset;
    3. generate multi-step forecasts either recursively or with direct horizon models.

    Why this is useful:
    reduction allows using strong tabular foundation models for forecasting while
    preserving sktime's forecasting interface and horizon/index semantics.

    Forecasting can be done via:

    - ``recursive`` strategy: fit a one-step model and iterate predictions forward.
    - ``direct`` strategy: fit one model per requested forecasting horizon step.

    The implementation intentionally separates model concerns from forecasting concerns
    through helper methods, so the structure can be re-used for other foundation
    tabular backends.

    Parameters
    ----------
    window_length : int, default=10
        Number of most recent observations used as model input features.
    strategy : {"recursive", "direct"}, default="recursive"
        Multi-step forecasting strategy.
    use_kv_cache : bool, default=True
        Whether to request key-value cache usage in the underlying model, if
        supported by the backend estimator.
    model_params : dict or None, default=None
        Parameters passed to ``TabICLRegressor`` during initialization.
    allow_exogenous : bool, default=False
        Reserved for future use and kept for API compatibility.
        Exogenous ``X`` is currently ignored.

    Notes
    -----
    - This estimator currently supports only positive, integer-valued,
      out-of-sample forecasting horizons.
    - ``X`` is accepted by API contract but ignored in the current implementation.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.advanced_tabicl import AdvancedTabICLForecaster
    >>> y = load_airline()
    >>> fh = ForecastingHorizon([1, 2, 3], is_relative=True)
    >>> forecaster = AdvancedTabICLForecaster(window_length=12)
    >>> forecaster.fit(y, fh=fh)  # doctest: +SKIP
    AdvancedTabICLForecaster(...)
    >>> y_pred = forecaster.predict(fh=fh)  # doctest: +SKIP
    """

    _tags = {
        # packaging
        "authors": ["sktime developers"],
        "python_dependencies": ["tabicl"],
        # estimator behavior
        "scitype:y": "univariate",
        "capability:multivariate": False,
        "capability:exogenous": False,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
    }

    _KV_CACHE_PARAM_CANDIDATES = (
        "use_kv_cache",
        "kv_cache",
        "enable_kv_cache",
    )

    def __init__(
        self,
        window_length: int = 10,
        strategy: str = "recursive",
        use_kv_cache: bool = True,
        model_params: dict | None = None,
        allow_exogenous: bool = False,
    ):
        self.window_length = window_length
        self.strategy = strategy
        self.use_kv_cache = use_kv_cache
        self.model_params = model_params
        self.allow_exogenous = allow_exogenous

        super().__init__()

        self.set_tags(**{"requires-fh-in-fit": self.strategy == "direct"})

    def _validate_relative_fh_steps(self, fh_relative):
        """Validate and return forecasting horizon steps as positive integers."""
        rel_values = fh_relative.to_numpy()
        rel_steps_float = np.asarray(rel_values, dtype=float).reshape(-1)

        if not np.all(np.isfinite(rel_steps_float)):
            raise ValueError("Forecasting horizon contains non-finite values.")

        # Allow values that are numerically integer-like, e.g., 2.0.
        rel_steps_rounded = np.rint(rel_steps_float)
        if not np.all(np.isclose(rel_steps_float, rel_steps_rounded)):
            raise ValueError(
                "AdvancedTabICLForecaster requires integer-valued relative "
                "forecasting horizon steps."
            )

        rel_steps = rel_steps_rounded.astype(int)
        if np.any(rel_steps <= 0):
            raise ValueError(
                "AdvancedTabICLForecaster supports only strictly positive "
                "out-of-sample forecasting horizons."
            )

        return rel_steps

    def _transform_series_to_tabular(self, y: pd.Series):
        """Convert univariate series to one-step tabular sliding-window dataset."""
        y_values = np.asarray(y.to_numpy(copy=True), dtype=float).reshape(-1)
        n_obs = y_values.shape[0]

        if n_obs <= self.window_length_:
            raise ValueError(
                "`window_length` must be smaller than number of observations in `y`."
            )

        n_rows = n_obs - self.window_length_
        X_tab = np.empty((n_rows, self.window_length_), dtype=float)
        y_tab = np.empty(n_rows, dtype=float)

        for i in range(n_rows):
            start = i
            stop = i + self.window_length_
            X_tab[i, :] = y_values[start:stop]
            y_tab[i] = y_values[stop]

        return X_tab, y_tab

    def _make_direct_training_data(self, y_values: np.ndarray, horizon: int):
        """Create tabular training data for direct forecasting at a fixed horizon."""
        if horizon <= 0:
            raise ValueError("All forecasting horizon steps must be positive.")

        n_obs = y_values.shape[0]
        n_rows = n_obs - self.window_length_ - horizon + 1
        if n_rows <= 0:
            raise ValueError(
                f"Not enough observations to fit direct model for horizon={horizon}."
            )

        X_tab = np.empty((n_rows, self.window_length_), dtype=float)
        y_tab = np.empty(n_rows, dtype=float)

        for i in range(n_rows):
            start = i
            stop = i + self.window_length_
            X_tab[i, :] = y_values[start:stop]
            y_tab[i] = y_values[stop + horizon - 1]

        return X_tab, y_tab

    def _get_model_class(self):
        """Return the model class used by this forecaster.

        Separated into a helper to make future backend swapping straightforward.
        """
        _check_soft_dependencies("tabicl", obj=self)
        from tabicl import TabICLRegressor

        return TabICLRegressor

    def _get_model_init_kwargs(self):
        """Return validated model init kwargs copied from ``model_params``."""
        if self.model_params is None:
            return {}

        if not isinstance(self.model_params, Mapping):
            raise TypeError("`model_params` must be a mapping (e.g., dict) or None.")

        return deepcopy(dict(self.model_params))

    def _init_model(self):
        """Initialize the underlying TabICL model from validated parameters."""
        model_class = self._get_model_class()
        model_kwargs = self._get_model_init_kwargs()
        model = model_class(**model_kwargs)
        model = self._handle_kv_cache(model)
        return model

    def _handle_kv_cache(self, model):
        """Apply KV cache preference to model when backend supports it.

        Writes introspection state for reproducible behavior diagnostics:
        ``kv_cache_requested_``, ``kv_cache_applied_``, ``kv_cache_field_``.
        """
        self.kv_cache_requested_ = bool(self.use_kv_cache)
        self.kv_cache_applied_ = False
        self.kv_cache_field_ = None

        if hasattr(model, "get_params") and hasattr(model, "set_params"):
            try:
                model_params = model.get_params(deep=False)
            except TypeError:
                model_params = model.get_params()

            for param_name in self._KV_CACHE_PARAM_CANDIDATES:
                if param_name in model_params:
                    model.set_params(**{param_name: self.use_kv_cache})
                    self.kv_cache_applied_ = True
                    self.kv_cache_field_ = param_name
                    return model

        for attr_name in self._KV_CACHE_PARAM_CANDIDATES:
            if hasattr(model, attr_name):
                setattr(model, attr_name, self.use_kv_cache)
                self.kv_cache_applied_ = True
                self.kv_cache_field_ = attr_name
                return model

        return model

    def _recursive_forecast(self, last_window: np.ndarray, fh):
        """Iteratively produce recursive forecasts over ``fh``."""
        rel_steps = self._validate_relative_fh_steps(fh)

        max_step = int(np.max(rel_steps))
        history = np.asarray(last_window, dtype=float).reshape(-1).copy()
        preds_by_step = {}

        for step in range(1, max_step + 1):
            # Roll one step ahead and append prediction to history for recursion.
            X_next = history[-self.window_length_ :].reshape(1, -1)
            y_next = self.model_.predict(X_next)
            y_next = float(np.asarray(y_next).reshape(-1)[0])
            preds_by_step[step] = y_next
            history = np.append(history, y_next)

        return np.asarray([preds_by_step[int(step)] for step in rel_steps], dtype=float)

    def _fit(self, y, X, fh):
        """Fit the forecaster to a univariate series."""
        # Exogenous X is accepted by interface but intentionally ignored for now.
        _ = X

        if not isinstance(self.window_length, int) or self.window_length < 1:
            raise ValueError("`window_length` must be a positive integer.")

        if self.strategy not in {"recursive", "direct"}:
            raise ValueError("`strategy` must be either 'recursive' or 'direct'.")

        self.window_length_ = self.window_length
        self.y_name_ = y.name

        y_values = np.asarray(y.to_numpy(copy=True), dtype=float).reshape(-1)
        self._y_values_ = y_values.copy()
        self.last_window_ = y_values[-self.window_length_ :].copy()

        self.model_ = None
        self.direct_models_ = None
        self.direct_horizons_ = None

        if self.strategy == "recursive":
            X_train, y_train = self._transform_series_to_tabular(y)
            self.model_ = self._init_model()
            self.model_.fit(X_train, y_train)
        else:
            if fh is None:
                raise ValueError(
                    "When `strategy='direct'`, `fh` must be provided in fit so "
                    "horizon-specific models can be trained."
                )

            fh_relative = fh.to_relative(self.cutoff)
            fh_steps = self._validate_relative_fh_steps(fh_relative)

            unique_horizons = sorted(set(int(h) for h in fh_steps))
            self.direct_horizons_ = unique_horizons
            self.direct_models_ = {}

            for horizon in unique_horizons:
                # Train one model per horizon step in direct mode.
                X_h, y_h = self._make_direct_training_data(y_values, horizon)
                model_h = self._init_model()
                model_h.fit(X_h, y_h)
                self.direct_models_[horizon] = model_h

        return self

    def _predict(self, fh, X):
        """Forecast future values for the requested forecasting horizon."""
        # Exogenous X is accepted by interface but intentionally ignored for now.
        _ = X

        fh_relative = fh.to_relative(self.cutoff)
        fh_steps = self._validate_relative_fh_steps(fh_relative)

        if self.strategy == "recursive":
            if self.model_ is None:
                raise RuntimeError(
                    "Recursive model is not initialized. Fit with "
                    "`strategy='recursive'` before predicting."
                )
            y_pred_values = self._recursive_forecast(self.last_window_, fh_relative)
        else:
            if self.direct_models_ is None:
                raise RuntimeError(
                    "Direct models were not initialized. Fit with `strategy='direct'` "
                    "and a non-empty out-of-sample `fh`."
                )

            X_last = self.last_window_.reshape(1, -1)
            y_pred_values = np.empty(fh_steps.shape[0], dtype=float)

            for i, step in enumerate(fh_steps):
                # Non-consecutive horizons are supported by keyed lookup.
                model_h = self.direct_models_.get(int(step))
                if model_h is None:
                    raise ValueError(
                        f"No direct model is available for horizon step {int(step)}. "
                        "Re-fit with an `fh` that includes all requested steps."
                    )
                y_hat = model_h.predict(X_last)
                y_pred_values[i] = float(np.asarray(y_hat).reshape(-1)[0])

        absolute_index = fh.to_absolute(self.cutoff).to_pandas()
        y_pred = pd.Series(y_pred_values, index=absolute_index, name=self.y_name_)
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {
            "window_length": 5,
            "strategy": "recursive",
            "use_kv_cache": True,
            "model_params": {},
            "allow_exogenous": False,
        }
