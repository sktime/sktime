"""Shared base class for zero-shot foundation-model forecasters."""

from contextlib import contextmanager

import numpy as np
from sklearn.utils import check_random_state

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.foundation._cache import FOUNDATION_MODEL_CACHE
from sktime.forecasting.foundation._config import InferenceConfig
from sktime.forecasting.foundation._format import (
    format_point_result,
    format_quantile_result,
)
from sktime.forecasting.foundation._result import ForecastRequest, ForecastResult


class BaseFoundationForecaster(BaseForecaster):
    """Shared lifecycle for zero-shot foundation-model forecasters.

    Concrete forecasters implement ``_load_model`` and ``_inference``. They can
    additionally override ``_update_attrs_in_fit`` and ``_cache_key_extra`` for
    model-specific fitted state and loading parameters, respectively.
    """

    _tags = {
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path=None,
        tokenizer_path=None,
        revision=None,
        config=None,
        device=None,
        dtype=None,
        quantization_config=None,
        random_state=None,
        ignore_deps=None,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.revision = revision
        self.config = config
        self.device = device
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.random_state = random_state
        self.ignore_deps = ignore_deps

        super().__init__()

    def __dynamic_tags__(self):
        """Clear soft-dependency tags when dependency checks are disabled."""
        if self.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __post_init__(self):
        """Initialize normalized copies of shared constructor parameters."""
        self.random_state_ = check_random_state(self.random_state)
        self.config_ = {} if self.config is None else self.config.copy()
        self.device_ = self._resolve_device()
        self.dtype_ = self._resolve_dtype()

    def _fit(self, y, X=None, fh=None):
        """Store zero-shot context and load shared immutable model state."""
        self._update_attrs_in_fit(y=y, X=X, fh=fh)
        self.context_y_ = y.copy()
        self.context_X_ = None if X is None else X.copy()
        self.model_handle_ = self._get_or_load_model_handle()
        return self

    def _predict(self, fh, X=None):
        """Forecast point predictions from normalized foundation output."""
        result, request = self._foundation_predict(fh=fh, X=X)
        return format_point_result(result=result, request=request, y=self.context_y_)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Forecast quantiles from normalized foundation output."""
        result, request = self._foundation_predict(fh=fh, X=X, alpha=alpha)
        return format_quantile_result(
            result=result,
            request=request,
            y=self.context_y_,
            alpha=alpha,
        )

    def _foundation_predict(self, fh, X=None, alpha=None, coverage=None):
        """Run model-specific inference with shared horizon and reload setup."""
        if fh is None:
            fh = self.fh

        request = self._make_forecast_request(
            fh=fh,
            alpha=alpha,
            coverage=coverage,
        )
        pred_len = max(request.relative_fh)

        self.model_handle_ = self._get_or_load_model_handle()
        with self._inference_context(handle=self.model_handle_):
            result = self._inference(
                handle=self.model_handle_,
                context_y=self.context_y_,
                context_X=self.context_X_,
                future_X=X,
                pred_len=pred_len,
                fh=fh,
                alpha=request.alpha,
            )
        if not isinstance(result, ForecastResult):
            raise TypeError(
                f"{self.__class__.__name__}._inference must return ForecastResult, "
                f"but returned {type(result).__name__}."
            )
        return result, request

    def _make_forecast_request(self, fh, alpha=None, coverage=None):
        """Normalize an sktime forecasting horizon for backend adapters."""
        relative_fh = np.asarray(fh.to_relative(self.cutoff).to_pandas(), dtype=int)
        absolute_index = fh.to_absolute(self.cutoff).to_pandas()
        return ForecastRequest(
            relative_fh=tuple(int(value) for value in relative_fh),
            absolute_index=absolute_index,
            cutoff=self.cutoff,
            alpha=None if alpha is None else tuple(alpha),
            coverage=None if coverage is None else tuple(coverage),
            inference=InferenceConfig(random_state=self.random_state),
        )

    def _get_or_load_model_handle(self):
        """Return an existing or process-cached immutable model handle."""
        handle = getattr(self, "model_handle_", None)
        if handle is not None:
            return handle

        return FOUNDATION_MODEL_CACHE.get_or_load(
            key=self._get_unique_model_key(),
            loader=self._load_model,
        )

    @contextmanager
    def _inference_context(self, handle):
        """Set local Torch seed, eval mode, and inference mode."""
        import torch

        model = handle.model
        if hasattr(model, "eval"):
            model.eval()

        devices = []
        model_device = getattr(model, "device", None)
        if model_device is not None:
            model_device = torch.device(model_device)
            if model_device.type == "cuda":
                devices = [model_device.index or torch.cuda.current_device()]

        with torch.random.fork_rng(devices=devices):
            if self.random_state_ is not None:
                torch.manual_seed(self.random_state_)
            with torch.inference_mode():
                yield

    def _resolve_dtype(self):
        if self.dtype == "torch.bfloat16":
            import torch

            return torch.bfloat16

        return self.dtype

    def _resolve_device(self):
        """Resolve explicit, configured, or automatic device selection once."""
        if self.device != "auto":
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_unique_model_key(self):
        """Build a deterministic cache key from model-loading parameters."""
        key_items = {
            "class": self.__class__.__name__,
            "config": self.config_,
            "device": self.device_,
            "dtype": self.dtype_,
            "model_path": self.model_path,
            "quantization_config": self.quantization_config,
            "revision": self.revision,
            "tokenizer_path": self.tokenizer_path,
            "extra": self._cache_key_extra(),
        }
        return tuple(sorted(key_items.items()))

    @classmethod
    def _has_implementation_of(cls, method):
        """Respect probabilistic capability tags for shared quantile logic."""
        if method == "_predict_quantiles" and not cls.get_class_tag(
            "capability:pred_int", False
        ):
            return False
        return super()._has_implementation_of(method)

    def __getstate__(self):
        """Return pickle state without potentially unpickleable backend objects."""
        state = self.__dict__.copy()
        state["model_handle_"] = None
        return state

    def __setstate__(self, state):
        """Restore estimator state; the backend is reloaded on next prediction."""
        self.__dict__.update(state)

    def _update_attrs_in_fit(self, y, X, fh):
        """Update model-specific fitted attributes before loading the model."""

    def _load_model(self):
        """Load native backend state and return a ``ModelHandle``."""
        raise NotImplementedError

    def _inference(
        self,
        handle,
        context_y,
        context_X,
        future_X,
        pred_len,
        fh,
        alpha=None,
    ):
        """Run native inference and return a ``ForecastResult``."""
        raise NotImplementedError

    def _cache_key_extra(self):
        """Return estimator-specific model-loading cache-key components."""
        return ()
