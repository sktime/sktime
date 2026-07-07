"""Shared base class for foundation-model forecasters."""

from abc import ABC, abstractmethod

import numpy as np

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.foundation._cache import (
    FOUNDATION_MODEL_CACHE,
    _stable_repr,
)
from sktime.forecasting.foundation._config import (
    FineTuneConfig,
    InferenceConfig,
    ModelLoadConfig,
)
from sktime.forecasting.foundation._format import (
    coverage_to_alpha,
    format_interval_result,
    format_point_result,
    format_quantile_result,
)
from sktime.forecasting.foundation._hf import stable_peft_key, stable_quantization_key
from sktime.forecasting.foundation._result import (
    ForecastRequest,
    ForecastResult,
    ModelContext,
    ModelHandle,
)


class BaseFoundationForecaster(BaseForecaster, ABC):
    """Shared base class for pretrained/foundation forecasting models."""

    def _fit(self, y, X=None, fh=None):
        """Fit foundation forecaster state around a loaded model handle."""
        self._init_foundation_runtime()

        if getattr(self, "model_handle_", None) is None:
            self.model_handle_ = self._get_or_load_model(for_training=False)

        self.context_ = self._make_context(
            y=y,
            X=X,
            fh=fh,
            cutoff=self.cutoff,
            handle=self.model_handle_,
        )

        if self._fine_tune_config.strategy != "zero-shot":
            training_data = self._make_training_data(
                y=y,
                X=X,
                fh=fh,
                purpose="fit",
                handle=self.model_handle_,
            )
            self.model_handle_ = self._train_model(
                handle=self.model_handle_,
                training_data=training_data,
                tune=self._fine_tune_config,
            )

        return self

    def _pretrain(self, y, X=None, fh=None):
        """Pretrain mutable foundation-model state."""
        self._init_foundation_runtime()

        if not self._foundation_spec.supports_pretrain:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support pretrain"
            )

        handle = getattr(self, "model_handle_", None)
        if handle is None:
            handle = self._get_or_load_model(for_training=True)

        training_data = self._make_training_data(
            y=y,
            X=X,
            fh=fh,
            purpose="pretrain",
            handle=handle,
        )
        self.model_handle_ = self._train_model(
            handle=handle,
            training_data=training_data,
            tune=self._fine_tune_config,
        )
        self.pretrained_artifact_ = self.model_handle_.metadata.get("artifact")
        return self

    def _pretrain_update(self, y, X=None, fh=None):
        """Update pretrained foundation-model state."""
        return self._pretrain(y=y, X=X, fh=fh)

    def _predict(self, fh, X=None):
        """Forecast point predictions from normalized foundation output."""
        result = self._foundation_predict(fh=fh, X=X, alpha=None, coverage=None)
        return self._format_point_result(result=result, fh=fh)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Forecast quantiles from normalized foundation output."""
        result = self._foundation_predict(fh=fh, X=X, alpha=alpha, coverage=None)
        return self._format_quantile_result(result=result, fh=fh, alpha=alpha)

    def _predict_interval(self, fh, X=None, coverage=None):
        """Forecast intervals from normalized foundation output."""
        alpha = coverage_to_alpha(coverage)
        result = self._foundation_predict(
            fh=fh,
            X=X,
            alpha=alpha,
            coverage=coverage,
        )
        return self._format_interval_result(
            result=result,
            fh=fh,
            coverage=coverage,
        )

    def _init_foundation_runtime(self):
        """Initialize normalized runtime config objects."""
        self._load_config = self._get_model_load_config()
        self._fine_tune_config = self._get_fine_tune_config()
        self._inference_config = self._get_inference_config()

    def _get_or_load_model(self, *, for_training: bool) -> ModelHandle:
        """Load a model handle, using the shared cache when allowed."""
        key = self._foundation_cache_key()
        shareable = self._is_cache_shareable(for_training=for_training)

        return FOUNDATION_MODEL_CACHE.get_or_load(
            key=key,
            loader=lambda: self._load_model(
                load=self._load_config,
                tune=self._fine_tune_config,
            ),
            shareable=shareable,
        )

    def _foundation_cache_key(self) -> tuple:
        """Return the shared cache key for this foundation-model load."""
        return (
            self._foundation_spec.family,
            self._load_config.model_path,
            self._load_config.revision,
            self._load_config.cache_dir,
            self._load_config.local_files_only,
            _stable_repr(self._load_config.device_map),
            _stable_repr(self._load_config.dtype),
            stable_quantization_key(self._load_config.quantization_config),
            stable_peft_key(self._fine_tune_config.parameter_efficient),
            _stable_repr(self._load_config.extra_load_kwargs),
            _stable_repr(self._cache_key_extra()),
        )

    def _is_cache_shareable(self, *, for_training: bool) -> bool:
        """Return whether this load should use shared immutable cache state."""
        if for_training:
            return False
        return self._fine_tune_config.strategy == "zero-shot"

    def _foundation_predict(self, fh, X=None, alpha=None, coverage=None):
        """Run a model-native prediction and return normalized output."""
        self._ensure_foundation_model_loaded()

        request = self._make_forecast_request(fh=fh, alpha=alpha, coverage=coverage)

        context = self.context_
        if X is not None:
            context = self._update_context_with_X(
                context=context,
                X=X,
                request=request,
            )

        return self._predict_native(
            handle=self.model_handle_,
            context=context,
            request=request,
        )

    def _ensure_foundation_model_loaded(self):
        """Reload model state lazily after pickle or cache release."""
        if getattr(self, "model_handle_", None) is None:
            self._init_foundation_runtime()
            self.model_handle_ = self._get_or_load_model(for_training=False)

    def _make_forecast_request(self, fh, alpha=None, coverage=None):
        """Normalize sktime forecasting inputs for model-specific hooks."""
        relative_fh = np.asarray(fh.to_relative(self.cutoff).to_pandas(), dtype=int)
        absolute_index = fh.to_absolute(self.cutoff).to_pandas()
        return ForecastRequest(
            relative_fh=tuple(int(value) for value in relative_fh),
            absolute_index=absolute_index,
            cutoff=self.cutoff,
            alpha=None if alpha is None else tuple(alpha),
            coverage=None if coverage is None else tuple(coverage),
            inference=self._inference_config,
        )

    def _format_point_result(self, result: ForecastResult, fh):
        """Format point forecasts with sktime output conventions."""
        request = self._make_forecast_request(fh=fh)
        return format_point_result(result=result, request=request, y=self._y)

    def _format_quantile_result(self, result: ForecastResult, fh, alpha):
        """Format quantile forecasts with sktime output conventions."""
        request = self._make_forecast_request(fh=fh, alpha=alpha)
        return format_quantile_result(
            result=result,
            request=request,
            y=self._y,
            alpha=alpha,
        )

    def _format_interval_result(self, result: ForecastResult, fh, coverage):
        """Format interval forecasts with sktime output conventions."""
        request = self._make_forecast_request(fh=fh, coverage=coverage)
        return format_interval_result(
            result=result,
            request=request,
            y=self._y,
            coverage=coverage,
        )

    def __getstate__(self):
        """Return pickle state without loaded model handles."""
        state = self.__dict__.copy()
        state["model_handle_"] = None
        return state

    def __setstate__(self, state):
        """Restore estimator state from pickle."""
        self.__dict__.update(state)

    @abstractmethod
    def _get_model_load_config(self) -> ModelLoadConfig:
        """Return normalized model loading policy from estimator params."""

    @abstractmethod
    def _load_model(
        self,
        load: ModelLoadConfig,
        tune: FineTuneConfig,
    ) -> ModelHandle:
        """Load or initialize model state."""

    @abstractmethod
    def _make_context(self, y, X, fh, cutoff, handle: ModelHandle) -> ModelContext:
        """Convert fitted sktime data into model-native prediction context."""

    @abstractmethod
    def _predict_native(
        self,
        handle: ModelHandle,
        context: ModelContext,
        request: ForecastRequest,
    ) -> ForecastResult:
        """Run model-native forecasting and return normalized output."""

    def _get_fine_tune_config(self) -> FineTuneConfig:
        """Return normalized tuning policy."""
        return FineTuneConfig(strategy="zero-shot")

    def _get_inference_config(self) -> InferenceConfig:
        """Return normalized inference policy."""
        return InferenceConfig()

    def _make_training_data(self, y, X, fh, purpose, handle: ModelHandle):
        """Build model-native training data."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training"
        )

    def _train_model(self, handle: ModelHandle, training_data, tune: FineTuneConfig):
        """Train or fine-tune model-native state."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support training"
        )

    def _update_context_with_X(
        self,
        context: ModelContext,
        X,
        request: ForecastRequest,
    ) -> ModelContext:
        """Update prediction context with future exogenous data."""
        return context

    def _cache_key_extra(self) -> tuple:
        """Return estimator-specific cache-key components."""
        return ()
