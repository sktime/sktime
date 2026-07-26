"""Concrete catalgue classes for forecasting experiments."""

from sktime.catalogues.forecasting.dummy import DummyForecastingCatalogue
from sktime.catalogues.forecasting.M4 import (
    M4CompetitionCatalogueDaily,
    M4CompetitionCatalogueHourly,
    M4CompetitionCatalogueMonthly,
    M4CompetitionCatalogueQuarterly,
    M4CompetitionCatalogueWeekly,
    M4CompetitionCatalogueYearly,
)

__all__ = [
    "DummyForecastingCatalogue",
    "M4CompetitionCatalogueHourly",
    "M4CompetitionCatalogueDaily",
    "M4CompetitionCatalogueMonthly",
    "M4CompetitionCatalogueQuarterly",
    "M4CompetitionCatalogueWeekly",
    "M4CompetitionCatalogueYearly",
]
