# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""TimesFM 3 forecaster for ``sktime``.

Wraps Google Research TimesFM 3.0 via the upstream ``timesfm3`` package.
"""

__all__ = ["TimesFM3Forecaster"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.singleton import _multiton

_RESERVED_CONFIG_KEYS = frozenset({"checkpoint_path", "device", "per_core_batch_size"})
_LICENSE_URL = "https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE"


class TimesFM3Forecaster(BaseForecaster):
    """Interface to Google TimesFM 3 zero-shot forecaster.

    TimesFM 3 is a pretrained multivariate time series foundation model
    supporting native joint forecasting of multiple targets with past-only
    and past-and-future covariates. See [1]_ and [2]_ for details.

    Exogenous variables are supplied via ``X`` in ``fit`` and ``predict``.
    Columns listed in ``past_covariates`` are treated as past-only covariates
    (known only over the historical context). All other ``X`` columns supplied
    in ``fit`` are treated as past-and-future covariates and must also be
    provided in ``predict`` for every step ``1 .. max(fh)`` ahead of the
    cutoff.

    Point forecasts use the upstream median quantile. Probabilistic forecasts
    are available through ``predict_quantiles`` for quantile levels
    ``0.1, 0.2, ..., 0.9``.

    Parameters
    ----------
    model_path : str, default="google/timesfm-3.0-pytorch"
        Hugging Face repository id or local checkpoint path for TimesFM 3.
    device : str or None, default=None
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``. If ``None``,
        upstream selects CUDA when available, otherwise CPU.
    batch_size : int, default=4
        Batch size passed to upstream ``ModelConfig.per_core_batch_size``.
    past_covariates : list of str or None, default=None
        Column names in ``X`` known only for the historical context window.
        Remaining ``X`` columns are treated as past-and-future covariates.
    config : dict or None, default=None
        Additional keyword arguments forwarded to upstream ``ModelConfig``.
        Reserved keys ``checkpoint_path``, ``device``, and
        ``per_core_batch_size`` must not appear here.
    use_symmetric_averaging : bool, default=False
        Whether to enable upstream symmetric averaging during inference.
    make_positive : bool, default=False
        Whether to clip forecasts to be non-negative when the context is
        non-negative.
    sort_quantiles : bool, default=True
        Whether to sort quantile outputs upstream before returning them.
    use_znorm : bool, default=False
        Whether to apply per-variate z-normalization upstream during inference.
    padding_mode : str, default="none"
        Upstream padding mode for past-and-future covariates. Supported values
        are ``"none"`` and ``"edge"``.
    license_accepted : bool, default=False
        Must be ``True`` to use the default pretrained weights, which are
        distributed under the TimesFM non-commercial license. Call
        ``TimesFM3Forecaster.print_license()`` for details.
    ignore_deps : bool, default=False
        If ``True``, skip soft-dependency checks (for testing).

    Attributes
    ----------
    forecaster_ : timesfm3.TimesFM3Forecaster
        Loaded upstream forecaster used for inference.

    References
    ----------
    .. [1] https://github.com/google-research/timesfm/
    .. [2] https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/

    Examples
    --------
    Univariate point forecast:

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.timesfm3 import TimesFM3Forecaster
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=12)
    >>> forecaster = TimesFM3Forecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP

    Multivariate forecast:

    >>> import pandas as pd
    >>> y_multi = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
    >>> forecaster = TimesFM3Forecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y_multi)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2])  # doctest: +SKIP

    Forecast with mixed past-only and past-and-future covariates:

    >>> y = pd.Series([1.0, 2.0, 3.0, 4.0])
    >>> X = pd.DataFrame({"past_only": [0.1, 0.2, 0.3, 0.4],
    ...                   "future_known": [1.0, 1.0, 1.0, 1.0]})
    >>> forecaster = TimesFM3Forecaster(
    ...     past_covariates=["past_only"], license_accepted=True
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y, X=X)  # doctest: +SKIP
    >>> X_future = pd.DataFrame({"future_known": [2.0, 2.0]})
    >>> y_pred = forecaster.predict(fh=[1, 2], X=X_future)  # doctest: +SKIP

    Quantile forecast:

    >>> y_quantiles = forecaster.predict_quantiles(
    ...     fh=[1, 2], alpha=[0.1, 0.5, 0.9]
    ... )  # doctest: +SKIP
    """

    _tags = {
        "authors": ["hasanfaesal"],
        "maintainers": ["hasanfaesal"],
        "python_dependencies": ["timesfm[torch]>=3.0.0,<4.0.0"],
        "capability:multivariate": True,
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:exogenous": True,
        "capability:categorical_in_X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:non_contiguous_X": False,
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_timesfm3"],
    }

    def __init__(
        self,
        model_path: str = "google/timesfm-3.0-pytorch",
        device: str | None = None,
        batch_size: int = 4,
        past_covariates: list[str] | None = None,
        config: dict | None = None,
        use_symmetric_averaging: bool = False,
        make_positive: bool = False,
        sort_quantiles: bool = True,
        use_znorm: bool = False,
        padding_mode: str = "none",
        license_accepted: bool = False,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.past_covariates = past_covariates
        self.config = config
        self.use_symmetric_averaging = use_symmetric_averaging
        self.make_positive = make_positive
        self.sort_quantiles = sort_quantiles
        self.use_znorm = use_znorm
        self.padding_mode = padding_mode
        self.license_accepted = license_accepted
        self.ignore_deps = ignore_deps

        self.forecaster_ = None

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters."""
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __getstate__(self):
        """Return state for pickling, excluding the unpickleable upstream model."""
        state = self.__dict__.copy()
        if hasattr(self, "forecaster_"):
            state["forecaster_"] = None
        return state

    def __setstate__(self, state):
        """Restore state from unpickled state dictionary."""
        self.__dict__.update(state)

    @classmethod
    def print_license(cls):
        """Print license information for TimesFM 3.0 pretrained weights."""
        print(
            "TimesFM 3.0 pretrained weights are distributed under the "
            "timesfm-non-commercial-license-v1.0, which restricts use to "
            "non-commercial, non-production scenarios."
        )
        print(f"Full license text: {_LICENSE_URL}")

    def _check_license(self):
        """Raise unless the user has accepted the TimesFM 3 weight license."""
        if not self.license_accepted:
            raise ValueError(
                "Use of TimesFM3Forecaster with pretrained weights is subject to "
                "the TimesFM non-commercial license. You must read and accept "
                "these terms to use the forecaster. To confirm acceptance, set "
                "the `license_accepted` parameter to True. To view the license, "
                f"call `TimesFM3Forecaster.print_license()` or visit {_LICENSE_URL}."
            )

    def _get_config_kwargs(self):
        """Build upstream ModelConfig keyword arguments."""
        cfg = {} if self.config is None else deepcopy(self.config)
        overlap = _RESERVED_CONFIG_KEYS.intersection(cfg)
        if overlap:
            reserved = ", ".join(sorted(overlap))
            raise ValueError(
                f"Reserved ModelConfig keys must not appear in `config`: {reserved}. "
                "Use the dedicated TimesFM3Forecaster constructor parameters instead."
            )
        cfg["checkpoint_path"] = self.model_path
        cfg["per_core_batch_size"] = self.batch_size
        if self.device is not None:
            cfg["device"] = self.device
        return cfg

    def _get_cache_key_kwargs(self):
        """Build deterministic cache key kwargs, omitting sensitive values."""
        cfg = self._get_config_kwargs()
        if "token" in cfg:
            cfg = deepcopy(cfg)
            cfg["token"] = "<redacted>"
        return cfg

    def _get_unique_key(self):
        """Build cache key for the multiton model loader."""
        return str(sorted(self._get_cache_key_kwargs().items()))

    def _load_model(self):
        """Load or retrieve the cached upstream TimesFM 3 forecaster."""
        if hasattr(self, "forecaster_") and self.forecaster_ is not None:
            return self.forecaster_

        self.forecaster_ = _CachedTimesFM3(
            key=self._get_unique_key(),
            config_kwargs=self._get_config_kwargs(),
        ).load()
        return self.forecaster_

    def _ensure_model_loaded(self):
        """Reload upstream forecaster if needed after unpickling."""
        if not hasattr(self, "forecaster_") or self.forecaster_ is None:
            if hasattr(self, "_is_fitted") and self._is_fitted:
                self.forecaster_ = self._load_model()

    def _get_max_variates(self):
        """Return maximum supported target plus covariate variates."""
        forecaster = self._load_model()
        return forecaster.model.transformer_config.transformer.max_variates

    def _validate_past_covariates(self, x_columns):
        """Validate ``past_covariates`` against available ``X`` columns."""
        if self.past_covariates is None:
            return []
        if len(self.past_covariates) != len(set(self.past_covariates)):
            raise ValueError(
                "`past_covariates` must contain unique column names, "
                f"but got duplicates in {self.past_covariates}."
            )
        unknown = set(self.past_covariates) - set(x_columns)
        if unknown:
            raise ValueError(
                "`past_covariates` contains columns not present in fit-time `X`: "
                f"{sorted(unknown)}."
            )
        return list(self.past_covariates)

    @staticmethod
    def _validate_numeric_exog(X, label):
        """Ensure exogenous columns are numeric."""
        non_numeric = [
            col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])
        ]
        if non_numeric:
            raise ValueError(
                f"{label} must contain numeric columns only; "
                f"non-numeric columns found: {non_numeric}."
            )

    def _partition_exog_columns(self, x_columns):
        """Split fit-time ``X`` columns into past-only and past-future groups."""
        past_only = self._validate_past_covariates(x_columns)
        past_future = [col for col in x_columns if col not in past_only]
        return past_only, past_future

    def _validate_variate_limit(self, n_targets, n_past_only, n_past_future):
        """Raise if total variates exceed the loaded model capacity."""
        total = n_targets + n_past_only + n_past_future
        limit = self._get_max_variates()
        if total > limit:
            raise ValueError(
                "Total number of target and covariate variates exceeds the "
                f"TimesFM 3 model limit: total={total}, limit={limit}. "
                "Reduce the number of target columns and/or exogenous columns."
            )

    def _truncate_context(self, y, X):
        """Return trailing context windows for ``y`` and optional ``X``."""
        forecaster = self._load_model()
        max_len = forecaster.global_context
        y_ctx = y.iloc[-max_len:] if len(y) > max_len else y
        if X is None:
            return y_ctx, None
        x_ctx = X.loc[y_ctx.index]
        return y_ctx, x_ctx

    def _fit(self, y, X, fh):
        """Fit forecaster to training data."""
        self._check_license()
        self._load_model()

        if self.past_covariates is not None and X is None:
            raise ValueError(
                "`past_covariates` were specified but no exogenous `X` was "
                "provided in fit."
            )

        if X is not None:
            self._validate_numeric_exog(X, "fit-time `X`")

        y_ctx, x_ctx = self._truncate_context(y, X)
        past_only_cols, past_future_cols = self._partition_exog_columns(
            [] if x_ctx is None else list(x_ctx.columns)
        )

        self._validate_variate_limit(
            n_targets=y_ctx.shape[1],
            n_past_only=len(past_only_cols),
            n_past_future=len(past_future_cols),
        )

        self._context_ = y_ctx
        self._past_only_cols_ = past_only_cols
        self._past_future_cols_ = past_future_cols
        self._y_index_names = y.index.names
        return self

    def _build_future_exog(self, X, horizon):
        """Validate and return prediction-time exogenous data."""
        if not self._past_future_cols_:
            if X is not None:
                raise ValueError(
                    "Exogenous `X` was provided in predict but no past-and-future "
                    "covariates were supplied in fit."
                )
            return None

        if X is None:
            raise ValueError(
                "Past-and-future covariates were supplied in fit, so exogenous `X` "
                f"must also be provided in predict for steps 1..{horizon}."
            )

        self._validate_numeric_exog(X, "prediction-time `X`")

        missing = set(self._past_future_cols_) - set(X.columns)
        if missing:
            raise ValueError(
                "Prediction-time `X` is missing past-and-future covariate columns "
                f"seen in fit: {sorted(missing)}."
            )

        extra = set(X.columns) - set(self._past_future_cols_)
        if extra:
            raise ValueError(
                "Prediction-time `X` contains columns that were not declared as "
                f"past-and-future covariates in fit: {sorted(extra)}."
            )

        if len(X) < horizon:
            raise ValueError(
                f"Prediction-time `X` must cover at least {horizon} future steps "
                f"ahead of the cutoff, but only {len(X)} rows were provided."
            )

        return X.iloc[:horizon]

    def _to_upstream_arrays(self, horizon, X_future):
        """Convert stored context and exogenous data to upstream numpy arrays."""
        target = self._context_.values.T.astype(np.float32)

        past_only = None
        if self._past_only_cols_:
            past_only = self._X.loc[
                self._context_.index, self._past_only_cols_
            ].values.T
            past_only = past_only.astype(np.float32)

        past_future = None
        if self._past_future_cols_:
            past = self._X.loc[self._context_.index, self._past_future_cols_].values.T
            future = X_future[self._past_future_cols_].values.T
            past_future = np.concatenate([past, future], axis=1).astype(np.float32)

        return target, past_only, past_future

    def _run_forecast(self, fh, X, return_quantiles):
        """Run upstream inference and return raw output plus index helpers."""
        self._ensure_model_loaded()
        forecaster = self._load_model()

        horizon = int(max(fh.to_relative(self.cutoff)))
        X_future = self._build_future_exog(X, horizon)
        target, past_only, past_future = self._to_upstream_arrays(horizon, X_future)

        output = forecaster.predict(
            context=target,
            horizon=horizon,
            past_only_covariates=past_only,
            past_future_covariates=past_future,
            return_quantiles=return_quantiles,
            use_symmetric_averaging=self.use_symmetric_averaging,
            make_positive=self.make_positive,
            sort_quantiles=self.sort_quantiles,
            use_znorm=self.use_znorm,
            padding_mode=self.padding_mode,
        )

        index = (
            ForecastingHorizon(range(1, horizon + 1)).to_absolute(self._cutoff)._values
        )
        pred_out = fh.get_expected_pred_idx(self._context_.values.T, cutoff=self.cutoff)
        return output, index, pred_out, horizon

    def _predict(self, fh, X):
        """Forecast time series at future horizon."""
        output, index, pred_out, _ = self._run_forecast(fh, X, return_quantiles=False)

        forecast = np.asarray(output.forecast)
        if forecast.ndim == 1:
            forecast = forecast.reshape(1, -1)

        pred_df = pd.DataFrame(
            forecast.T,
            index=index,
            columns=self._get_varnames(),
        )
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return prediction quantiles for a forecast."""
        output, index, pred_out, _ = self._run_forecast(fh, X, return_quantiles=True)

        available = [round(q, 3) for q in self.forecaster_.config.quantiles]
        alpha_rounded = [round(a, 3) for a in alpha]
        if not set(alpha_rounded).issubset(set(available)):
            raise ValueError(
                "Requested quantiles are not all available in the TimesFM 3 "
                f"checkpoint: requested={alpha_rounded}, available={available}."
            )

        quantiles = np.asarray(output.quantiles)
        if quantiles.ndim == 2:
            quantiles = quantiles[np.newaxis, :, :]

        var_names = self._get_varnames()
        quantile_indices = [available.index(a) for a in alpha_rounded]

        columns = pd.MultiIndex.from_product([var_names, alpha])
        data = {}
        for s, var in enumerate(var_names):
            for a, idx in zip(alpha, quantile_indices):
                data[(var, a)] = quantiles[s, :, idx]

        pred_df = pd.DataFrame(data, index=index, columns=columns)
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {"license_accepted": True, "device": "cpu"},
            {
                "license_accepted": True,
                "device": "cpu",
                "past_covariates": [],
            },
        ]


@_multiton
class _CachedTimesFM3:
    """Multiton-backed cache wrapper for a loaded TimesFM 3 forecaster."""

    def __init__(self, key, config_kwargs):
        self.key = key
        self.config_kwargs = config_kwargs
        self.forecaster = None

    def load(self):
        """Load upstream forecaster if needed and return cached instance."""
        if self.forecaster is not None:
            return self.forecaster

        from timesfm3 import ModelConfig
        from timesfm3 import TimesFM3Forecaster as _UpstreamForecaster

        config = ModelConfig(**self.config_kwargs)
        self.forecaster = _UpstreamForecaster(config=config)
        return self.forecaster
