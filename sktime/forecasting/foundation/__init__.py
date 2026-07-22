"""Internal foundation-model forecasting infrastructure."""

from sktime.forecasting.foundation._base import BaseFoundationForecaster
from sktime.forecasting.foundation._cache import clear_foundation_model_cache
from sktime.forecasting.foundation._result import (
    ForecastRequest,
    ForecastResult,
    ModelHandle,
)

__all__ = [
    "BaseFoundationForecaster",
    "ForecastRequest",
    "ForecastResult",
    "ModelHandle",
    "clear_foundation_model_cache",
]
