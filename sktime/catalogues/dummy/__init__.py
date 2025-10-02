"""Dummy catalogues."""

from sktime.catalogues.dummy._dummy_classification_catalogue import (
    DummyClassificationCatalogue,
)
from sktime.catalogues.dummy._dummy_forecasting_catalogue import (
    DummyForecastingCatalogue,
)

__all__ = [
    "DummyClassificationCatalogue",
    "DummyForecastingCatalogue",
]
