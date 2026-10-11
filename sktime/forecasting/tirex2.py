# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TiRex-2 forecaster."""

__author__ = [
    "yash-sangwan",
    "danilyef",
    "lukfischer",
    "Tigxy",
    "Suad0",
    "danieleb1861",
    "martinloretzzz",
]
__all__ = ["TiRex2Forecaster"]

import warnings

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.dependencies import _safe_import
from sktime.utils.singleton import _multiton

torch = _safe_import("torch")

# devices accepted by tirex2.load_model, plus sktime's "auto" resolution
_VALID_DEVICES = ("auto", "cpu", "cuda", "mps")


def _to_tensor(df):
    """Convert a ``(T, V)`` frame to a 2D ``(V, T)`` float32 tensor, or None."""
    if df is None:
        return None
    return torch.as_tensor(df.to_numpy(dtype="float32").T, dtype=torch.float32)


def _resolve_device():
    """Return the best available device among ``cuda``, ``mps``, ``cpu``."""
    from skbase.utils.dependencies import _check_soft_dependencies

    if _check_soft_dependencies("torch", severity="none"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


@_multiton
class _CachedTiRex2:
    """Cached TiRex-2 model, shared across forecasters with the same key.

    TiRex-2 is zero-shot and immutable at inference time, so sharing one
    instance across estimators with identical load settings has no side effects.
    """

    def __init__(self, key, load_kwargs):
        self.key = key
        self.load_kwargs = load_kwargs
        self.model = None

    def load_from_checkpoint(self):
        """Load the TiRex-2 model, reusing the cached instance if present."""
        if self.model is not None:
            return self.model

        from tirex2 import load_model

        self.model = load_model(**self.load_kwargs)
        return self.model


class TiRex2Forecaster(BaseForecaster):
    """Interface to the TiRex-2 zero-shot forecaster by NX-AI.

    TiRex-2 is a pretrained xLSTM-based time series foundation model for
    zero-shot forecasting. It natively supports multivariate targets, past
    covariates, and known-future covariates, and outputs nine quantile levels
    (0.1 to 0.9) per prediction step.

    The model is run in eager mode, via ``torch.compiler.set_stance``, because
    the underlying package applies ``torch.compile`` unconditionally, which
    requires a working C++ toolchain and fails on machines without one.

    Parameters
    ----------
    model_path : str, default="NX-AI/TiRex-2"
        Hugging Face repo id or local checkpoint directory.

        The decontaminated variants ``NX-AI/TiRex-2-gifteval-zs``,
        ``NX-AI/TiRex-2-gifteval-pretrain`` and ``NX-AI/TiRex-2-fevbench``
        are gated on Hugging Face and require authentication to download.
    device : {"auto", "cpu", "cuda", "mps"}, default="auto"
        Device used for inference. ``"auto"`` resolves to ``cuda``, then
        ``mps``, then ``cpu``, depending on availability.
    revision : str, optional, default=None
        Model repository revision, branch, tag, or commit.
    tta_sign_flip : bool, optional, default=None
        Sign-flip test-time augmentation. ``None`` uses the checkpoint default.
        Roughly doubles inference cost when enabled.
    tta_diff : bool, optional, default=None
        Postprocessor differencing. ``None`` uses the checkpoint default.
    hf_kwargs : dict, optional, default=None
        Additional keyword arguments passed to ``huggingface_hub.snapshot_download``.
    ignore_deps : bool, default=False
        If True, soft dependency checks are skipped. Intended for tests and
        controlled environments.

    References
    ----------
    .. [1] https://github.com/NX-AI/tirex-2
    .. [2] TiRex-2: Generalizing TiRex to Multivariate Data and Streaming,
       arXiv:2607.01204

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tirex2 import TiRex2Forecaster
    >>> y = load_airline()
    >>> forecaster = TiRex2Forecaster()  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "yash-sangwan",
            "danilyef",
            "lukfischer",
            "Tigxy",
            "Suad0",
            "danieleb1861",
            "martinloretzzz",
        ],
        "maintainers": ["yash-sangwan"],
        "python_version": ">=3.11",
        "python_dependencies": ["tirex-2", "torch"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:insample": False,
        "capability:categorical_in_X": False,
        "capability:non_contiguous_X": False,
        "requires-fh-in-fit": False,
        # CI and test flags
        # -----------------
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_tirex2"],
    }

    def __init__(
        self,
        model_path: str = "NX-AI/TiRex-2",
        device: str = "auto",
        revision: str = None,
        tta_sign_flip: bool = None,
        tta_diff: bool = None,
        hf_kwargs: dict = None,
        ignore_deps: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.revision = revision
        self.tta_sign_flip = tta_sign_flip
        self.tta_diff = tta_diff
        self.hf_kwargs = hf_kwargs
        self.ignore_deps = ignore_deps

        self.model = None

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        self._check_device_param()

    def _check_device_param(self):
        """Validate the ``device`` parameter."""
        if self.device not in _VALID_DEVICES:
            raise ValueError(
                f"Error in {type(self).__name__}, device must be one of "
                f"{_VALID_DEVICES}, but found {self.device!r}."
            )

    def _get_device(self):
        """Return the resolved runtime device, resolving ``auto`` on first use.

        Resolution is lazy rather than done in ``__post_init__`` alone, because
        ``BaseForecaster`` skips ``__post_init__`` when the soft dependencies
        are not present in the environment.
        """
        if getattr(self, "_device", None) is None:
            self._check_device_param()
            if self.device == "auto":
                self._device = _resolve_device()
            else:
                self._device = self.device
        return self._device

    def _get_hf_kwargs(self):
        """Return the keyword arguments forwarded to ``snapshot_download``."""
        hf_kwargs = {} if self.hf_kwargs is None else dict(self.hf_kwargs)
        if self.revision is not None:
            hf_kwargs.setdefault("revision", self.revision)
        return hf_kwargs

    def __getstate__(self):
        """Return state for pickling, excluding the unpickleable model."""
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        """Restore state from an unpickled state dictionary."""
        self.__dict__.update(state)

    def _get_load_kwargs(self):
        """Return the keyword arguments used to load the model."""
        kwargs = {"ckpt_path": self.model_path, "device": self._get_device()}
        hf_kwargs = self._get_hf_kwargs()
        if hf_kwargs:
            kwargs["hf_kwargs"] = hf_kwargs
        return kwargs

    def _get_unique_model_key(self):
        """Return the cache key identifying an interchangeable loaded model.

        Only load-affecting settings take part. Prediction-time options such as
        ``tta_sign_flip`` and ``tta_diff`` do not change the loaded weights and
        are therefore excluded.
        """
        kwargs = self._get_load_kwargs()
        return str(sorted(kwargs.items(), key=lambda item: item[0]))

    def _load_model(self):
        """Load the TiRex-2 model through the process-local cache."""
        return _CachedTiRex2(
            key=self._get_unique_model_key(),
            load_kwargs=self._get_load_kwargs(),
        ).load_from_checkpoint()

    def _ensure_model_loaded(self):
        """Reload the model if it is missing, e.g. after unpickling."""
        if getattr(self, "model", None) is None:
            self.model = self._load_model()
        return self.model

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        TiRex-2 is zero-shot, so no training happens. This loads the checkpoint
        and stores the most recent context window used at predict time.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series, one column per target variate.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series, covering at least the index of ``y``.
        fh : ForecastingHorizon, optional (default=None)
            Not required in fit.

        Returns
        -------
        self : reference to self
        """
        model = self._ensure_model_loaded()

        # the model keeps only the most recent context_len steps internally,
        # so truncate here to bound memory and make the behaviour explicit
        self._context = y.iloc[-model.context_len :]
        self._y_columns = y.columns
        self._y_index_names = y.index.names

        if X is not None:
            self._X_context = X.reindex(self._context.index)
            self._X_columns = list(X.columns)
        else:
            self._X_context = None
            self._X_columns = []

        return self

    def _split_covariates(self, X_future, pred_len):
        """Split exogenous data into past-only and known-future covariates.

        Columns seen in both ``fit`` and ``predict`` are treated as known-future
        covariates, and are returned as their past values followed by their
        future values, which is the layout TiRex-2 expects. Columns seen only in
        ``fit`` are treated as past-only covariates.

        Parameters
        ----------
        X_future : pd.DataFrame or None
            Exogenous data passed to ``predict``, covering the forecast horizon.
        pred_len : int
            Dense horizon length, i.e. ``max(fh)``.

        Returns
        -------
        past_cov : pd.DataFrame or None
            Past-only covariates, indexed like the stored context.
        future_cov : pd.DataFrame or None
            Known-future covariates, of length ``len(context) + pred_len``.
        """
        cls_name = type(self).__name__

        if self._X_context is None:
            if X_future is not None:
                raise ValueError(
                    f"Error in {cls_name}: X was passed to predict but not to "
                    "fit. To use exogenous data, pass X in fit as well."
                )
            return None, None

        if X_future is None:
            return self._X_context, None

        unknown = [col for col in X_future.columns if col not in self._X_columns]
        if unknown:
            raise ValueError(
                f"Error in {cls_name}: X passed to predict has columns not seen "
                f"in fit: {unknown}. Columns seen in fit were {self._X_columns}."
            )

        if len(X_future) < pred_len:
            raise ValueError(
                f"Error in {cls_name}: X passed to predict must cover the full "
                f"forecasting horizon, but has {len(X_future)} rows while "
                f"{pred_len} are required."
            )

        future_cols = [col for col in self._X_columns if col in X_future.columns]
        past_cols = [col for col in self._X_columns if col not in X_future.columns]

        if future_cols:
            future_cov = pd.concat(
                [self._X_context[future_cols], X_future[future_cols].iloc[:pred_len]]
            )
        else:
            future_cov = None

        past_cov = self._X_context[past_cols] if past_cols else None
        return past_cov, future_cov

    def _build_timeseries(self, X_future, pred_len):
        """Assemble the ``TimeseriesType`` instance passed to the model.

        Parameters
        ----------
        X_future : pd.DataFrame or None
            Exogenous data passed to ``predict``.
        pred_len : int
            Dense horizon length, i.e. ``max(fh)``.

        Returns
        -------
        timeseries : tirex2.TimeseriesType
        """
        from tirex2 import TimeseriesType

        past_cov, future_cov = self._split_covariates(X_future, pred_len)

        return TimeseriesType(
            target=_to_tensor(self._context),
            past_covariates=_to_tensor(past_cov),
            future_covariates=_to_tensor(future_cov),
        )

    def _forecast_raw(self, fh, X):
        """Run the model once and return raw quantile forecasts.

        This is the only place ``model.forecast`` is called. The call runs under
        ``torch.compiler.set_stance("force_eager")``, because ``tirex-2`` applies
        ``torch.compile`` unconditionally on its forward path, which requires a
        C++ toolchain and fails on machines that do not have one.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data covering the forecast horizon.

        Returns
        -------
        values : np.ndarray, shape (n_targets, n_quantiles, pred_len)
            Quantile forecasts over the dense horizon ``1 .. max(fh)``. Steps
            past the model's maximum supported horizon are filled with ``nan``.
        levels : np.ndarray, shape (n_quantiles,)
            Native quantile levels of the model.
        """
        model = self._ensure_model_loaded()
        pred_len = int(max(fh.to_relative(self.cutoff)))
        timeseries = self._build_timeseries(X, pred_len)

        predict_kwargs = {}
        if self.tta_sign_flip is not None:
            predict_kwargs["tta_sign_flip"] = self.tta_sign_flip
        if self.tta_diff is not None:
            predict_kwargs["tta_diff"] = self.tta_diff

        with torch.compiler.set_stance("force_eager"):
            forecast = model.forecast(
                [timeseries],
                prediction_length=pred_len,
                output_type="numpy",
                **predict_kwargs,
            )

        values = forecast[0]

        # the model caps the horizon at future_len and only logs a warning,
        # so pad the tail to keep the returned index consistent with fh
        n_returned = values.shape[-1]
        if n_returned < pred_len:
            warnings.warn(
                f"{type(self).__name__} was asked to forecast {pred_len} steps "
                f"ahead, but {self.model_path} supports at most {n_returned}. "
                f"Forecasts beyond step {n_returned} are returned as nan.",
                stacklevel=2,
            )
            pad_shape = values.shape[:-1] + (pred_len - n_returned,)
            pad = np.full(pad_shape, np.nan, dtype=values.dtype)
            values = np.concatenate([values, pad], axis=-1)

        levels = np.asarray([round(float(q), 6) for q in model.quantiles])
        return values, levels

    def _predict_native_quantile_grid(self, fh, X):
        """Forecast, and return the native quantile grid on the requested ``fh``.

        Shared by ``_predict``, ``_predict_quantiles`` and ``_predict_proba``.
        The model is queried over the dense horizon ``1 .. max(fh)``, and the
        result is then subset to the steps actually requested.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame or None
            Exogenous data covering the forecast horizon.

        Returns
        -------
        levels : np.ndarray, shape (n_quantiles,)
            Native quantile levels of the model, ascending.
        q_values : np.ndarray, shape (n_targets, n_quantiles, n_fh)
            Quantile forecasts at the requested horizon steps.
        pred_index : pd.Index
            Absolute index expected in the returned forecasts.
        var_names : pd.Index
            Column names of ``y`` as seen in ``fit``.
        """
        values, levels = self._forecast_raw(fh, X)

        # values cover the dense horizon 1 .. max(fh), so step k sits at k - 1
        rel_idx = fh.to_relative(self.cutoff).to_numpy().astype(int) - 1
        q_values = values[:, :, rel_idx]

        pred_index = fh.to_absolute(self.cutoff).to_pandas()
        pred_index.names = self._y_index_names

        return levels, q_values, pred_index, self._y_columns

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional (default=None)
            Exogenous data covering the forecast horizon.

        Returns
        -------
        y_pred : pd.DataFrame
            Point forecasts, one column per target variate.
        """
        levels, q_values, pred_index, var_names = self._predict_native_quantile_grid(
            fh, X
        )

        # TiRex-2 emits quantiles only, so the median is the point forecast
        median_idx = int(np.argmin(np.abs(levels - 0.5)))
        point = q_values[:, median_idx, :]

        return pd.DataFrame(point.T, index=pred_index, columns=var_names)

    def _predict_quantiles(self, fh, X, alpha):
        """Compute/return quantile forecasts.

        Requested levels are linearly interpolated onto the model's native
        quantile grid. ``np.interp`` saturates outside the grid, so levels
        beyond the native range are clamped to the nearest native quantile.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional (default=None)
            Exogenous data covering the forecast horizon.
        alpha : list of float
            Probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has a multi-index: first level is the variable name from
            ``y`` in fit, second level are the values of ``alpha``.
            Row index is ``fh``. Entries are quantile forecasts.
        """
        levels, q_values, pred_index, var_names = self._predict_native_quantile_grid(
            fh, X
        )

        # interpolate along the quantile axis, giving (n_targets, n_alpha, n_fh)
        interpolated = np.apply_along_axis(
            lambda col: np.interp(alpha, levels, col), 1, q_values
        )

        columns = pd.MultiIndex.from_product([var_names, alpha])
        values = interpolated.reshape(len(var_names) * len(alpha), -1).T
        return pd.DataFrame(values, index=pred_index, columns=columns)

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        Returns a ``skpro`` ``HistogramQPD`` built from the model's native
        quantile grid. ``tails="mass"`` places the remaining tail mass as point
        masses at the outermost native quantiles, matching ``FlowStateForecaster``.
        As the native grid spans 0.1 to 0.9, this puts 10% of the mass in an atom
        at each end, so the distribution is mixed rather than continuous.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional (default=None)
            Exogenous data covering the forecast horizon.
        marginal : bool, optional (default=True)
            Whether the returned distribution is marginal by time index.

        Returns
        -------
        pred_dist : skpro BaseDistribution
            Predictive distribution, with same index and columns as ``_predict``.
        """
        from skpro.distributions import HistogramQPD

        levels, q_values, pred_index, var_names = self._predict_native_quantile_grid(
            fh, X
        )

        # HistogramQPD expects rows indexed by (quantile level, time), so move
        # the quantile axis first and flatten it together with the time axis
        stacked = np.transpose(q_values, (1, 2, 0)).reshape(
            len(levels) * len(pred_index), len(var_names)
        )
        row_index = pd.MultiIndex.from_product([levels, pred_index])
        quantile_df = pd.DataFrame(stacked, index=row_index, columns=var_names)

        return HistogramQPD(
            quantile_df, tails="mass", index=pred_index, columns=var_names
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Both parameter sets use the default, ungated checkpoint and differ only
        in ``tta_diff``, which is a prediction-time option. They therefore share
        one cache entry, and the checkpoint is downloaded only once per session.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"``
            set. There are currently no reserved values for forecasters.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {"model_path": "NX-AI/TiRex-2", "device": "cpu"}
        params2 = {"model_path": "NX-AI/TiRex-2", "device": "cpu", "tta_diff": False}
        return [params1, params2]
