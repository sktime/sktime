"""Collections of estimators, datasets, and metrics."""

from sktime.catalogues.classification import DummyClassificationCatalogue
from sktime.catalogues.forecasting import (
    DummyForecastingCatalogue,
    M4CompetitionCatalogue,
)

__all__ = [
    "DummyClassificationCatalogue",
    "DummyForecastingCatalogue",
    "M4CompetitionCatalogue",
]
