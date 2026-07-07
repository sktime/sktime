"""Internal foundation-model forecasting infrastructure."""

from sktime.forecasting.foundation._base import BaseFoundationForecaster
from sktime.forecasting.foundation._cache import (
    FOUNDATION_MODEL_CACHE,
    FoundationModelCache,
    clear_foundation_model_cache,
    foundation_model_cache_info,
)
from sktime.forecasting.foundation._config import (
    FineTuneConfig,
    FoundationModelSpec,
    InferenceConfig,
    ModelLoadConfig,
    ParameterEfficientTuneConfig,
)
from sktime.forecasting.foundation._result import (
    ForecastRequest,
    ForecastResult,
    ModelContext,
    ModelHandle,
)
from sktime.forecasting.foundation._testing import FakeFoundationForecaster

__all__ = [
    "BaseFoundationForecaster",
    "FOUNDATION_MODEL_CACHE",
    "FakeFoundationForecaster",
    "FineTuneConfig",
    "ForecastRequest",
    "ForecastResult",
    "FoundationModelCache",
    "FoundationModelSpec",
    "InferenceConfig",
    "ModelContext",
    "ModelHandle",
    "ModelLoadConfig",
    "ParameterEfficientTuneConfig",
    "clear_foundation_model_cache",
    "foundation_model_cache_info",
]
