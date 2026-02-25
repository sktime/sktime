"""Collections of estimators, datasets, and metrics."""

from sktime.catalogues.classification import (
    BakeOffCatalogue,
    DummyClassificationCatalogue,
)
from sktime.catalogues.forecasting import DummyForecastingCatalogue

__all__ = [
    BakeOffCatalogue,
    DummyClassificationCatalogue,
    DummyForecastingCatalogue,
]
