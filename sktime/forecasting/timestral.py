# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Timestral forecaster for ``sktime``.

Wraps Mistral AI Timestral foundation model.
"""

__all__ = ["TimestralForecaster"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.singleton import _multiton

_LICENSE_URL = "https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE"
_NATIVE_QUANTILES = (
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.99,
)


def _to_numpy(x):
    """Convert tensor or array-like to CPU numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TimestralForecaster(BaseForecaster):
    """Interface to Mistral AI Timestral zero-shot forecaster.

    Timestral is an open-weights time series foundation model family developed
    by Mistral AI, trained using Decoupled Horizon Forecasting (DHF). It
    supports native multivariate forecasting, past-only and known-future covariates,
    categorical covariates, missing values, and probabilistic forecasting with 21
    quantiles.

    Parameters
    ----------
    model_path : str, default="mistralai/Timestral-209M"
        Hugging Face repository id or local directory containing model weights and
        config.
    device : str or None, default=None
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.
    revision : str or None, default=None
        Hugging Face Hub branch, tag or commit hash to load weights from.
    local_files_only : bool, default=False
        Whether to read only from the local cache without network access.
    max_context : int or None, default=None
        Maximum context window length. Defaults to the model trained context length
        (up to 32,768).
    symmetric_averaging : bool, default=True
        Whether to enable upstream test-time symmetric reflection augmentation.
    clip_nonnegative : bool, default=True
        Whether to clamp forecasts to 0 for series with strictly non-negative context.
    license_accepted : bool, default=False
        Must be ``True`` to use pretrained weights distributed under Mistral AI's
        Modified MIT license. Call ``TimestralForecaster.print_license()`` for details.
    ignore_deps : bool, default=False
        If ``True``, skip soft-dependency checks (for testing).

    References
    ----------
    .. [1] https://github.com/mistralai/timestral
    .. [2] https://mistral.ai/news/timestral/

    Examples
    --------
    Univariate point forecast:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timestral import TimestralForecaster
    >>> y = load_airline().iloc[:-12]
    >>> forecaster = TimestralForecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Multivariate forecast:

    >>> import pandas as pd
    >>> y_multi = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    >>> forecaster = TimestralForecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y_multi)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2])  # doctest: +SKIP

    Quantile forecast:

    >>> y_quantiles = forecaster.predict_quantiles(
    ...     fh=[1, 2], alpha=[0.1, 0.5, 0.9]
    ... )  # doctest: +SKIP
    """

    _tags = {
        "category": ["deep_learning", "foundation_model"],
        "authors": ["Artjom Joosen"],
        "maintainers": ["hasanfaesal"],
        "python_dependencies": [
            "torch>=2.4.0",
            "huggingface_hub>=0.23",
            "safetensors>=0.4",
        ],
        "capability:multivariate": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": True,
        "capability:categorical_in_X": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "tests:vm": True,
        "tests:libs": ["sktime.libs.timestral"],
        "tests:specific": ["sktime.forecasting.tests.test_timestral"],
    }

    def __init__(
        self,
        model_path: str = "mistralai/Timestral-209M",
        device: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        max_context: int | None = None,
        symmetric_averaging: bool = True,
        clip_nonnegative: bool = True,
        license_accepted: bool = False,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.revision = revision
        self.local_files_only = local_files_only
        self.max_context = max_context
        self.symmetric_averaging = symmetric_averaging
        self.clip_nonnegative = clip_nonnegative
        self.license_accepted = license_accepted
        self.ignore_deps = ignore_deps

        self.forecaster = None

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters."""
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __getstate__(self):
        """Return state for pickling, excluding the unpickleable upstream model."""
        state = self.__dict__.copy()
        if hasattr(self, "forecaster"):
            state["forecaster"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    @classmethod
    def print_license(cls):
        """Print license information for Timestral pretrained weights."""
        print(
            "Timestral pretrained weights are distributed under the Mistral Modified "
            "MIT license, which requires a commercial license agreement if the global "
            "consolidated monthly revenue exceeds $20M for the preceding month."
        )
        print(f"Full license text: {_LICENSE_URL}")

    def _check_license(self):
        """Raise unless the user has accepted the Timestral weight license."""
        if not self.license_accepted:
            raise ValueError(
                "Use of TimestralForecaster with pretrained weights is subject to "
                "the Mistral Modified MIT license (requiring a commercial agreement "
                "if global consolidated monthly revenue exceeds $20M). To confirm "
                "acceptance, set the `license_accepted` parameter to True. To "
                f"view the license, call `TimestralForecaster.print_license()` or "
                f"visit {_LICENSE_URL}."
            )

    def _get_load_kwargs(self):
        """Build upstream from_pretrained loading arguments."""
        kwargs = {
            "path_or_repo_id": self.model_path,
            "local_files_only": self.local_files_only,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self.revision is not None:
            kwargs["revision"] = self.revision
        return kwargs

    def _get_cache_key_kwargs(self):
        """Build deterministic cache key kwargs, omitting sensitive values."""
        cfg = self._get_load_kwargs()
        if "token" in cfg:
            cfg = deepcopy(cfg)
            cfg["token"] = "<redacted>"
        return cfg

    def _get_unique_key(self):
        """Build cache key for the multiton model loader."""
        return str(sorted(self._get_cache_key_kwargs().items()))

    def _load_model(self):
        """Load or retrieve the cached upstream Timestral forecaster."""
        if hasattr(self, "forecaster") and self.forecaster is not None:
            return self.forecaster

        self.forecaster = _CachedTimestral(
            key=self._get_unique_key(),
            load_kwargs=self._get_load_kwargs(),
        ).load()
        return self.forecaster

    def _ensure_model_loaded(self):
        """Reload upstream forecaster if needed after unpickling."""
        if not hasattr(self, "forecaster") or self.forecaster is None:
            if hasattr(self, "_is_fitted") and self._is_fitted:
                self.forecaster = self._load_model()

    def _partition_exog_columns(self, X):
        """Split fit-time ``X`` columns into numeric/categorical and past/known groups.

        Columns present in fit-time ``X`` that also appear in predict-time ``X``
        are treated as known covariates (their future values are known across
        the horizon). Columns present only in fit-time ``X`` are past-only covariates.
        Each group is divided into numeric and categorical subsets.
        """
        if self._cur_X is None:
            if X is not None:
                raise ValueError(
                    "Exogenous `X` was provided in predict but none was provided "
                    "in fit."
                )
            return [], [], [], []

        fit_cols = list(self._cur_X.columns)
        predict_cols = [] if X is None else list(X.columns)
        unknown = set(predict_cols) - set(fit_cols)
        if unknown:
            raise ValueError(
                "Prediction-time `X` contains columns not seen in fit-time `X`: "
                f"{sorted(unknown)}."
            )

        known_cols = [col for col in fit_cols if col in predict_cols]
        past_cols = [col for col in fit_cols if col not in predict_cols]

        def is_cat(col):
            s = self._cur_X[col]
            return s.dtype.name in ["category", "object", "string"]

        cat_past = [col for col in past_cols if is_cat(col)]
        num_past = [col for col in past_cols if not is_cat(col)]
        cat_known = [col for col in known_cols if is_cat(col)]
        num_known = [col for col in known_cols if not is_cat(col)]

        return num_past, num_known, cat_past, cat_known

    def _build_future_exog(self, X, known_cols, horizon):
        """Validate and return prediction-time known covariates."""
        if not known_cols:
            return None

        if len(X) < horizon:
            raise ValueError(
                f"Prediction-time `X` must cover at least {horizon} future steps "
                f"ahead of the cutoff, but only {len(X)} rows were provided."
            )

        return X[known_cols].iloc[:horizon]

    def _encode_categorical(self, series_fit, series_future=None):
        """Encode categorical series to float array with codes, preserving NaNs."""
        combined = (
            series_fit
            if series_future is None
            else pd.concat([series_fit, series_future])
        )
        cat = pd.Categorical(combined)
        codes = cat.codes.astype(np.float32)
        # pd.Categorical codes use -1 for missing values; map to np.nan
        codes[cat.codes == -1] = np.nan
        if series_future is None:
            return codes
        n_fit = len(series_fit)
        return codes[:n_fit], codes[n_fit:]

    def _to_upstream_arrays(
        self,
        y_ctx,
        num_past_cols,
        num_known_cols,
        cat_past_cols,
        cat_known_cols,
        X_future,
    ):
        """Convert context and exogenous data to upstream 3D numpy arrays.

        Upstream Timestral expects (batch, variates, time) tensors.
        """
        # Target shape: (1, V, T)
        target = y_ctx.values.T.astype(np.float32)[np.newaxis, ...]

        # Numeric past covariates: (1, C_num_past, T)
        num_past = None
        if num_past_cols:
            vals = self._cur_X.loc[y_ctx.index, num_past_cols].values.T.astype(
                np.float32
            )
            num_past = vals[np.newaxis, ...]

        # Numeric known covariates: (1, C_num_known, T + horizon)
        num_known = None
        if num_known_cols:
            past_vals = self._cur_X.loc[y_ctx.index, num_known_cols].values.T.astype(
                np.float32
            )
            future_vals = X_future[num_known_cols].values.T.astype(np.float32)
            concat_vals = np.concatenate([past_vals, future_vals], axis=1)
            num_known = concat_vals[np.newaxis, ...]

        # Categorical past covariates: (1, C_cat_past, T)
        cat_past = None
        if cat_past_cols:
            past_list = []
            for col in cat_past_cols:
                s_fit = self._cur_X.loc[y_ctx.index, col]
                past_list.append(self._encode_categorical(s_fit))
            cat_past = np.stack(past_list, axis=0)[np.newaxis, ...]

        # Categorical known covariates: (1, C_cat_known, T + horizon)
        cat_known = None
        if cat_known_cols:
            known_list = []
            for col in cat_known_cols:
                s_fit = self._cur_X.loc[y_ctx.index, col]
                s_fut = X_future[col]
                codes_all = self._encode_categorical(s_fit, s_fut)
                concat_codes = (
                    np.concatenate(codes_all)
                    if isinstance(codes_all, tuple)
                    else codes_all
                )
                known_list.append(concat_codes)
            cat_known = np.stack(known_list, axis=0)[np.newaxis, ...]

        return target, num_past, num_known, cat_past, cat_known

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        self._check_license()
        self._load_model()
        self._cur_y = y
        self._cur_X = X
        self._y_index_names = y.index.names
        return self

    def _update(self, y, X=None, update_params=True):
        """Update forecaster with new training data."""
        from sktime.datatypes import update_data

        self._cur_y = update_data(self._cur_y, y)
        if X is not None:
            self._cur_X = update_data(self._cur_X, X) if self._cur_X is not None else X
        return self

    def _run_forecast(self, fh, X=None):
        """Run upstream Timestral inference and return raw forecast object plus indexes.

        Returns forecast object, absolute forecast index, expected prediction index,
        and horizon integer.
        """
        self._check_license()
        self._ensure_model_loaded()
        forecaster = self._load_model()

        horizon = int(max(fh.to_relative(self.cutoff)))
        if horizon <= 0:
            raise ValueError(
                f"Horizon must be strictly positive (out-of-sample), but got {horizon}."
            )

        # Determine context window length
        trained_len = forecaster.max_context
        if self.max_context is not None:
            limit = min(trained_len, self.max_context)
        else:
            limit = trained_len

        y_ctx = self._cur_y.iloc[-limit:] if len(self._cur_y) > limit else self._cur_y

        # Partition exogenous variables
        (
            num_past_cols,
            num_known_cols,
            cat_past_cols,
            cat_known_cols,
        ) = self._partition_exog_columns(X)
        all_known = num_known_cols + cat_known_cols
        X_future = self._build_future_exog(X, all_known, horizon)

        (
            target,
            num_past,
            num_known,
            cat_past,
            cat_known,
        ) = self._to_upstream_arrays(
            y_ctx,
            num_past_cols,
            num_known_cols,
            cat_past_cols,
            cat_known_cols,
            X_future,
        )

        forecast = forecaster.forecast(
            context=target,
            prediction_length=horizon,
            past_covariates=num_past,
            known_covariates=num_known,
            categorical_past_covariates=cat_past,
            categorical_known_covariates=cat_known,
            symmetric_averaging=self.symmetric_averaging,
            clip_nonnegative=self.clip_nonnegative,
        )

        index = (
            ForecastingHorizon(range(1, horizon + 1)).to_absolute(self._cutoff)._values
        )
        pred_out = fh.get_expected_pred_idx(self._cur_y, cutoff=self.cutoff)
        return forecast, index, pred_out, horizon

    def _predict(self, fh, X=None):
        """Forecast time series at specified future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None, default=None
            Exogenous data.

        Returns
        -------
        y_pred : pd.DataFrame
            Point forecasts corresponding to ``fh``, using upstream median.
        """
        forecast, index, pred_out, _ = self._run_forecast(fh, X)

        # forecast.median shape: (1, V, horizon) or (V, horizon)
        median = _to_numpy(forecast.median)
        if median.ndim == 3:
            median = median[0]  # (V, horizon)
        elif median.ndim == 1:
            median = median[np.newaxis, :]

        pred_df = pd.DataFrame(
            median.T,
            index=index,
            columns=self._get_varnames(),
        )
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        Requested levels are linearly interpolated on the 21 native quantile levels.
        Levels beyond the native range [0.01, 0.99] are clamped to the nearest
        native quantile.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None, default=None
            Exogenous data.
        alpha : list of float
            Quantile levels to return, each strictly between 0 and 1.

        Returns
        -------
        quantiles_df : pd.DataFrame
            Quantile forecasts with 2-level MultiIndex columns (variable, alpha).
        """
        forecast, index, pred_out, _ = self._run_forecast(fh, X)

        # forecast.quantiles shape: (1, V, horizon, 21) or (V, horizon, 21)
        quantiles = _to_numpy(forecast.quantiles)
        if quantiles.ndim == 4:
            quantiles = quantiles[0]  # (V, horizon, 21)
        elif quantiles.ndim == 2:
            quantiles = quantiles[np.newaxis, :, :]

        native_levels = np.asarray(_NATIVE_QUANTILES, dtype=float)

        # Interpolate along the quantile axis (axis=2)
        interpolated = np.apply_along_axis(
            lambda row: np.interp(alpha, native_levels, row), 2, quantiles
        )

        var_names = self._get_varnames()
        columns = pd.MultiIndex.from_product([var_names, alpha])
        # transposed to (horizon, V * len(alpha))
        values = interpolated.transpose(1, 0, 2).reshape(len(index), -1)
        pred_df = pd.DataFrame(values, index=index, columns=columns)
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    def _predict_proba(self, fh, X=None, marginal=True):
        """Compute/return a fully probabilistic forecast.

        Returns a ``skpro`` ``HistogramQPD`` built from the 21 native quantile levels.
        ``tails="mass"`` places residual tail probability as point masses at the
        outermost native quantiles (0.01 and 0.99), matching ``predict_quantiles``.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None, default=None
            Exogenous data.
        marginal : bool, default=True
            Whether to return marginal distributions.

        Returns
        -------
        dist : skpro.distributions.HistogramQPD
            Fitted quantile-parameterized distribution.
        """
        from skpro.distributions import HistogramQPD

        forecast, index, pred_out, _ = self._run_forecast(fh, X)

        # forecast.quantiles shape: (1, V, horizon, 21) or (V, horizon, 21)
        quantiles = _to_numpy(forecast.quantiles)
        if quantiles.ndim == 4:
            quantiles = quantiles[0]  # (V, horizon, 21)
        elif quantiles.ndim == 2:
            quantiles = quantiles[np.newaxis, :, :]

        levels = np.asarray(_NATIVE_QUANTILES, dtype=float)
        var_names = self._get_varnames()
        pred_index = pd.Index(index)
        pred_index.names = self._y_index_names

        mask = np.array([x in pred_out for x in pred_index], dtype=bool)
        sel_index = pred_index[mask]
        q_sel = quantiles[:, mask, :]

        # Format rows indexed by (quantile level, time), columns by target
        stacked = np.transpose(q_sel, (2, 1, 0)).reshape(
            len(levels) * len(sel_index), len(var_names)
        )
        row_index = pd.MultiIndex.from_product([levels, sel_index])
        quantile_df = pd.DataFrame(stacked, index=row_index, columns=var_names)

        return HistogramQPD(
            quantile_df, tails="mass", index=sel_index, columns=var_names
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for the estimator."""
        params1 = {
            "model_path": "mistralai/Timestral-5M",
            "license_accepted": True,
            "ignore_deps": True,
        }
        params2 = {
            "model_path": "mistralai/Timestral-5M",
            "symmetric_averaging": False,
            "clip_nonnegative": False,
            "license_accepted": True,
            "ignore_deps": True,
        }
        return [params1, params2]


@_multiton
class _CachedTimestral:
    """Multiton-backed cache wrapper for a loaded Timestral forecaster."""

    def __init__(self, key, load_kwargs):
        self.key = key
        self.load_kwargs = load_kwargs
        self.forecaster = None

    def load(self):
        """Load upstream forecaster if needed and return cached instance."""
        if self.forecaster is not None:
            return self.forecaster

        from sktime.libs.timestral import TimestralForecaster as _UpstreamForecaster

        self.forecaster = _UpstreamForecaster.from_pretrained(**self.load_kwargs)
        return self.forecaster
