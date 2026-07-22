"""Internal foundation-model forecasting infrastructure."""

from sktime.forecasting.foundation._base import BaseFoundationForecaster
from sktime.forecasting.foundation._cache import clear_foundation_model_cache
from sktime.forecasting.foundation._result import (
    ForecastRequest,
    ForecastResult,
    ModelHandle,
)
from sktime.forecasting.foundation._spec import FoundationModelSpec

__all__ = [
    "BaseFoundationForecaster",
    "ForecastRequest",
    "ForecastResult",
    "FoundationModelSpec",
    "ModelHandle",
    "clear_foundation_model_cache",
]
