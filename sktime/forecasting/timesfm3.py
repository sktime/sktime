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
    The split between past-only and past-and-future covariates is inferred
    from the data: fit-time ``X`` columns that are also present in the
    predict-time ``X`` are treated as past-and-future covariates (their future
    values are read from the predict-time ``X`` for every step
    ``1 .. max(fh)`` ahead of the cutoff); fit-time ``X`` columns absent from
    the predict-time ``X`` are treated as past-only covariates.

    Point forecasts use the upstream median quantile. Probabilistic forecasts
    are available through ``predict_quantiles``. Native checkpoint levels
    ``0.1, 0.2, ..., 0.9`` are returned exactly; intermediate levels are
    linearly interpolated, and levels outside that range are clamped to the
    nearest native quantile.

    Parameters
    ----------
    model_path : str, default="google/timesfm-3.0-pytorch"
        Hugging Face repository id or local checkpoint path for TimesFM 3.
    device : str or None, default=None
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``. If ``None``,
        upstream selects CUDA when available, otherwise CPU.
    batch_size : int, default=4
        Batch size passed to upstream ``ModelConfig.per_core_batch_size`` [3]_.
    config : dict or None, default=None
        Additional keyword arguments forwarded to upstream ``ModelConfig`` [3]_.
        Commonly useful keys include ``input_patch_length`` (int),
        ``output_patch_length`` (int), ``quantiles`` (list of float),
        ``use_stitching`` (bool), ``use_linear_detrending`` (bool),
        ``linear_detrending_threshold`` (float), ``use_iterative_cpm_revin``
        (bool), ``use_variate_attention`` (bool), ``use_sdpa`` (bool), and the
        Hugging Face Hub download keys ``cache_dir``, ``force_download``,
        ``token``, ``revision`` and ``local_files_only``. See [3]_ for the full
        list. The reserved keys ``checkpoint_path``, ``device`` and
        ``per_core_batch_size`` are set via the ``model_path``, ``device`` and
        ``batch_size`` parameters and must not appear here.
    use_symmetric_averaging : bool, default=False
        Whether to enable upstream symmetric averaging during inference.
    make_positive : bool, default=False
        Whether to clip forecasts to be non-negative when the context is
        non-negative.
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

    References
    ----------
    .. [1] https://github.com/google-research/timesfm/
    .. [2] https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
    .. [3] ``ModelConfig`` fields (upstream source):
       https://github.com/google-research/timesfm/blob/master/src/timesfm3/torch/timesfm3_forecaster.py

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

    Forecast with mixed past-only and past-and-future covariates. The split is
    inferred from the data: ``past_only`` appears only in the fit-time ``X``,
    while ``future_known`` appears in both the fit-time and predict-time ``X``:

    >>> y = pd.Series([1.0, 2.0, 3.0, 4.0])
    >>> X = pd.DataFrame({"past_only": [0.1, 0.2, 0.3, 0.4],
    ...                   "future_known": [1.0, 1.0, 1.0, 1.0]})
    >>> forecaster = TimesFM3Forecaster(license_accepted=True)  # doctest: +SKIP
    >>> forecaster.fit(y, X=X)  # doctest: +SKIP
    >>> X_future = pd.DataFrame({"future_known": [2.0, 2.0]})
    >>> y_pred = forecaster.predict(fh=[1, 2], X=X_future)  # doctest: +SKIP

    Quantile forecast:

    >>> y_quantiles = forecaster.predict_quantiles(
    ...     fh=[1, 2], alpha=[0.1, 0.5, 0.9]
    ... )  # doctest: +SKIP
    """

    _tags = {
        "authors": ["rajatsen91", "siriuz42", "hasanfaesal"],
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
        config: dict | None = None,
        use_symmetric_averaging: bool = False,
        make_positive: bool = False,
        use_znorm: bool = False,
        padding_mode: str = "none",
        license_accepted: bool = False,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.config = config
        self.use_symmetric_averaging = use_symmetric_averaging
        self.make_positive = make_positive
        self.use_znorm = use_znorm
        self.padding_mode = padding_mode
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
        if hasattr(self, "forecaster") and self.forecaster is not None:
            return self.forecaster

        self.forecaster = _CachedTimesFM3(
            key=self._get_unique_key(),
            config_kwargs=self._get_config_kwargs(),
        ).load()
        return self.forecaster

    def _ensure_model_loaded(self):
        """Reload upstream forecaster if needed after unpickling."""
        if not hasattr(self, "forecaster") or self.forecaster is None:
            if hasattr(self, "_is_fitted") and self._is_fitted:
                self.forecaster = self._load_model()

    def _get_max_variates(self):
        """Return maximum supported target plus covariate variates."""
        forecaster = self._load_model()
        return forecaster.model.transformer_config.transformer.max_variates

    def _partition_exog_columns(self, X):
        """Split fit-time ``X`` columns into past-only and past-future groups.

        The split is inferred from the data: fit-time columns that also appear
        in the predict-time ``X`` are past-and-future (their future values are
        known); fit-time columns absent from the predict-time ``X`` are
        past-only.
        """
        if self._X is None:
            if X is not None:
                raise ValueError(
                    "Exogenous `X` was provided in predict but none was provided "
                    "in fit."
                )
            return [], []
        fit_cols = list(self._X.columns)
        predict_cols = [] if X is None else list(X.columns)
        unknown = set(predict_cols) - set(fit_cols)
        if unknown:
            raise ValueError(
                "Prediction-time `X` contains columns not seen in fit-time `X`: "
                f"{sorted(unknown)}."
            )
        past_future = [col for col in fit_cols if col in predict_cols]
        past_only = [col for col in fit_cols if col not in predict_cols]
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

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Only loads the upstream model and stores ``y`` and ``X`` (retained by
        the base class). The past-only vs past-and-future covariate split is
        inferred at predict time from the columns supplied in the predict-time
        ``X``.
        """
        self._check_license()
        self._load_model()
        self._y_index_names = y.index.names
        return self

    def _build_future_exog(self, X, past_future_cols, horizon):
        """Validate and return prediction-time past-and-future covariates."""
        if not past_future_cols:
            return None

        if len(X) < horizon:
            raise ValueError(
                f"Prediction-time `X` must cover at least {horizon} future steps "
                f"ahead of the cutoff, but only {len(X)} rows were provided."
            )

        return X[past_future_cols].iloc[:horizon]

    def _to_upstream_arrays(self, y_ctx, past_only_cols, past_future_cols, X_future):
        """Convert context and exogenous data to upstream numpy arrays."""
        target = y_ctx.values.T.astype(np.float32)

        past_only = None
        if past_only_cols:
            past_only = self._X.loc[y_ctx.index, past_only_cols].values.T
            past_only = past_only.astype(np.float32)

        past_future = None
        if past_future_cols:
            past = self._X.loc[y_ctx.index, past_future_cols].values.T
            future = X_future[past_future_cols].values.T
            past_future = np.concatenate([past, future], axis=1).astype(np.float32)

        return target, past_only, past_future

    def _run_forecast(self, fh, X):
        """Run upstream inference and return raw output plus index helpers.

        Always requests quantiles so a single forward pass serves ``_predict``,
        ``_predict_quantiles`` and ``_predict_proba``. Quantiles are always
        sorted, since sktime promises a consistent output.
        """
        self._ensure_model_loaded()
        forecaster = self._load_model()

        horizon = int(max(fh.to_relative(self.cutoff)))

        max_len = forecaster.global_context
        y_ctx = self._y.iloc[-max_len:] if len(self._y) > max_len else self._y

        past_only_cols, past_future_cols = self._partition_exog_columns(X)
        self._validate_variate_limit(
            n_targets=y_ctx.shape[1],
            n_past_only=len(past_only_cols),
            n_past_future=len(past_future_cols),
        )

        X_future = self._build_future_exog(X, past_future_cols, horizon)
        target, past_only, past_future = self._to_upstream_arrays(
            y_ctx, past_only_cols, past_future_cols, X_future
        )

        output = forecaster.predict(
            context=target,
            horizon=horizon,
            past_only_covariates=past_only,
            past_future_covariates=past_future,
            return_quantiles=True,
            use_symmetric_averaging=self.use_symmetric_averaging,
            make_positive=self.make_positive,
            sort_quantiles=True,
            use_znorm=self.use_znorm,
            padding_mode=self.padding_mode,
        )

        index = (
            ForecastingHorizon(range(1, horizon + 1)).to_absolute(self._cutoff)._values
        )
        pred_out = fh.get_expected_pred_idx(target, cutoff=self.cutoff)
        return output, index, pred_out, horizon

    def _predict(self, fh, X):
        """Forecast time series at future horizon."""
        output, index, pred_out, _ = self._run_forecast(fh, X)

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
        """Compute/return prediction quantiles for a forecast.

        Requested levels are linearly interpolated onto the checkpoint's native
        quantile grid. ``np.interp`` saturates outside the grid, so levels
        beyond the native range are clamped to the nearest native quantile.
        """
        output, index, pred_out, _ = self._run_forecast(fh, X)

        available = np.asarray(self.forecaster.config.quantiles, dtype=float)
        quantiles = np.asarray(output.quantiles)
        if quantiles.ndim == 2:
            quantiles = quantiles[np.newaxis, :, :]

        interpolated = np.apply_along_axis(
            lambda row: np.interp(alpha, available, row), 2, quantiles
        )

        var_names = self._get_varnames()
        columns = pd.MultiIndex.from_product([var_names, alpha])
        values = interpolated.transpose(1, 0, 2).reshape(len(index), -1)
        pred_df = pd.DataFrame(values, index=index, columns=columns)
        pred_df.index.names = self._y_index_names

        dateindex = pred_df.index.get_level_values(-1).map(lambda x: x in pred_out)
        return pred_df.loc[dateindex]

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return a fully probabilistic forecast.

        Returns a ``skpro`` ``HistogramQPD`` built from the checkpoint's native
        quantile grid (``0.1, 0.2, ..., 0.9``). ``tails="mass"`` places the
        residual tail probability as point masses at the outermost native
        quantiles, matching the clamping behavior of ``predict_quantiles``.
        """
        from skpro.distributions import HistogramQPD

        output, index, pred_out, _ = self._run_forecast(fh, X)

        levels = np.asarray(self.forecaster.config.quantiles, dtype=float)
        quantiles = np.asarray(output.quantiles)
        if quantiles.ndim == 2:
            quantiles = quantiles[np.newaxis, :, :]
        # quantiles: (n_targets, n_fh, n_quantiles)

        var_names = self._get_varnames()
        pred_index = pd.Index(index)
        pred_index.names = self._y_index_names

        mask = np.array([x in pred_out for x in pred_index], dtype=bool)
        sel_index = pred_index[mask]
        q_sel = quantiles[:, mask, :]

        # HistogramQPD rows indexed by (quantile level, time), columns by target
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
        """Return testing parameter settings for the estimator."""
        return [
            {"license_accepted": True, "device": "cpu"},
            {"license_accepted": True, "device": "cpu", "make_positive": True},
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
