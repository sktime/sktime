"""Infrastructure for implementing foundation-model forecasters.

Contributor guide
-----------------
New zero-shot model adapters should subclass :class:`BaseFoundationForecaster`
and use the small data contracts exported here:

``FoundationModelSpec``
    Separates standard model identity/runtime settings from backend-specific
    loading and prediction keyword arguments.
``ModelHandle``
    Groups the expensive native objects that may be cached and shared.
``ForecastResult``
    Normalizes native model output before the base class creates sktime pandas
    output.

The two required subclass hooks are ``_load_model`` and ``_inference``. See the
``BaseFoundationForecaster`` class docstring and hook docstrings for the complete
lifecycle, shape conventions, cache requirements, and optional extension points.
"""

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
