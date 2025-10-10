"""Forecasting datasets."""

from .airline import Airline
from .hierarchical_sales_toydata import HierarchicalSalesToydata
from .longley import Longley
from .lynx import Lynx
from .m5_competition import M5Dataset
from .macroeconomic import Macroeconomic
from .monash._forecasting_data import ForecastingData
from .shampoo_sales import ShampooSales
from .solar import Solar
from .uschange import USChange

__all__ = [
    "Airline",
    "ForecastingData",
    "HierarchicalSalesToydata",
    "Longley",
    "Lynx",
    "Macroeconomic",
    "ShampooSales",
    "Solar",
    "USChange",
    "M5Dataset",
]
