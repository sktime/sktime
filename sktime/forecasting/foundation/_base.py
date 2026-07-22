"""Shared base class for zero-shot foundation-model forecasters."""

from contextlib import contextmanager
from dataclasses import replace

import numpy as np
from sklearn.utils import check_random_state

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.foundation._cache import FOUNDATION_MODEL_CACHE
from sktime.forecasting.foundation._format import (
    format_point_result,
    format_quantile_result,
)
from sktime.forecasting.foundation._result import ForecastRequest, ForecastResult
from sktime.forecasting.foundation._spec import FoundationModelSpec


class BaseFoundationForecaster(BaseForecaster):
    """Shared lifecycle for zero-shot foundation-model forecasters.

    Concrete forecasters implement ``_load_model`` and ``_inference``. They can
    additionally override ``_update_attrs_in_fit`` and ``_cache_key_extra`` for
    model-specific fitted state and loading parameters, respectively. Non-Torch
    backends set ``_uses_torch_inference_context`` to ``False``.
    """

    _tags = {
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "tests:vm": True,
    }
    _uses_torch_inference_context = True

    def __init__(self, model_spec: FoundationModelSpec):
        if not isinstance(model_spec, FoundationModelSpec):
            raise TypeError("model_spec must be a FoundationModelSpec")
        self.model_spec = model_spec
        super().__init__()

    def __dynamic_tags__(self):
        """Clear soft-dependency tags when dependency checks are disabled."""
        if self.model_spec.ignore_deps:
            self.set_tags(python_dependencies=[])

    def __post_init__(self):
        """Initialize normalized copies of shared constructor parameters."""
        spec = self.model_spec
        self.model_spec_ = replace(
            spec,
            config=self._resolve_config(spec.config),
            device=self._resolve_device(spec.device),
            dtype=self._resolve_dtype(spec.dtype),
            random_state=self._resolve_random_state(spec.random_state),
            load_extra_kwargs=spec.load_extra_kwargs,
            predict_extra_kwargs=spec.predict_extra_kwargs,
        )

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

    def _foundation_predict(self, fh, X=None, alpha=None):
        """Run model-specific inference with shared horizon and reload setup."""
        if fh is None:
            fh = self.fh

        request = self._make_forecast_request(fh=fh, alpha=alpha)
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

    def _make_forecast_request(self, fh, alpha=None):
        """Normalize an sktime forecasting horizon for backend adapters."""
        relative_fh = np.asarray(fh.to_relative(self.cutoff).to_pandas(), dtype=int)
        absolute_index = fh.to_absolute(self.cutoff).to_pandas()
        return ForecastRequest(
            relative_fh=tuple(int(value) for value in relative_fh),
            absolute_index=absolute_index,
            alpha=None if alpha is None else tuple(alpha),
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
        """Apply local Torch seeding, eval mode, and inference mode when enabled."""
        if not self._uses_torch_inference_context:
            yield
            return

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
            if self.model_spec_.random_state is not None:
                torch.manual_seed(self.model_spec_.random_state)
            with torch.inference_mode():
                yield

    def _resolve_random_state(self, random_state):
        rng = check_random_state(random_state)
        return (
            None if random_state is None else int(rng.randint(np.iinfo(np.int32).max))
        )

    def _resolve_config(self, config):
        if config is None:
            return None

        return config.copy()

    def _resolve_dtype(self, dtype):
        if dtype == "torch.bfloat16":
            import torch

            return torch.bfloat16

        return dtype

    def _resolve_device(self, device):
        """Resolve explicit, configured, or automatic device selection once."""
        if device != "auto":
            return device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_unique_model_key(self):
        """Build a deterministic cache key from model-loading parameters."""
        spec = self.model_spec_
        key_items = {
            "class": self.__class__.__name__,
            # for zero-shot forecasters, the cached model is indifferent to config
            # "config": spec.config,
            "device": spec.device,
            "dtype": spec.dtype,
            "model_path": spec.model_path,
            "quantization_config": spec.quantization_config,
            "revision": spec.revision,
            "tokenizer_path": spec.tokenizer_path,
            "load_extra_kwargs": spec.load_extra_kwargs,
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

    def _update_model_spec(self, **changes):
        """Update normalized runtime settings derived during fit or initialization."""
        self.model_spec_ = replace(self.model_spec_, **changes)
