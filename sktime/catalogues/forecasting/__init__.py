"""Concrete catalgue classes for forecasting experiments."""

from sktime.catalogues.forecasting.dummy import DummyForecastingCatalogue
from sktime.catalogues.forecasting.M4 import (
    M4CompetitionCatalogue,
)

__all__ = ["DummyForecastingCatalogue", "M4CompetitionCatalogue"]
