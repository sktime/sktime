"""M4 forecasting competition catalogue."""

from sktime.catalogues.forecasting.M4._daily import M4CompetitionCatalogueDaily
from sktime.catalogues.forecasting.M4._hourly import M4CompetitionCatalogueHourly
from sktime.catalogues.forecasting.M4._monthly import M4CompetitionCatalogueMonthly
from sktime.catalogues.forecasting.M4._quarterly import M4CompetitionCatalogueQuarterly
from sktime.catalogues.forecasting.M4._weekly import M4CompetitionCatalogueWeekly
from sktime.catalogues.forecasting.M4._yearly import M4CompetitionCatalogueYearly

__all__ = [
    "M4CompetitionCatalogueHourly",
    "M4CompetitionCatalogueDaily",
    "M4CompetitionCatalogueMonthly",
    "M4CompetitionCatalogueQuarterly",
    "M4CompetitionCatalogueWeekly",
    "M4CompetitionCatalogueYearly",
]
