"""Collections of estimators, datasets, and metrics."""

from sktime.catalogues.classification import DummyClassificationCatalogue
from sktime.catalogues.forecasting import (
    DummyForecastingCatalogue,
    M4CompetitionCatalogueDaily,
    M4CompetitionCatalogueHourly,
    M4CompetitionCatalogueMonthly,
    M4CompetitionCatalogueQuarterly,
    M4CompetitionCatalogueWeekly,
    M4CompetitionCatalogueYearly,
)

__all__ = [
    "DummyClassificationCatalogue",
    "DummyForecastingCatalogue",
    "M4CompetitionCatalogueHourly",
    "M4CompetitionCatalogueDaily",
    "M4CompetitionCatalogueMonthly",
    "M4CompetitionCatalogueQuarterly",
    "M4CompetitionCatalogueWeekly",
    "M4CompetitionCatalogueYearly",
]
