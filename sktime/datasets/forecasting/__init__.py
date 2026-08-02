"""Forecasting datasets."""

from sktime.datasets.forecasting.airline import Airline
from sktime.datasets.forecasting.hierarchical_sales_toydata import (
    HierarchicalSalesToydata,
)
from sktime.datasets.forecasting.longley import Longley
from sktime.datasets.forecasting.lynx import Lynx
from sktime.datasets.forecasting.m5_competition import M5Dataset
from sktime.datasets.forecasting.macroeconomic import Macroeconomic
from sktime.datasets.forecasting.monash._forecasting_data import ForecastingData
from sktime.datasets.forecasting.pbs import PBS
from sktime.datasets.forecasting.shampoo_sales import ShampooSales
from sktime.datasets.forecasting.solar import Solar
from sktime.datasets.forecasting.uschange import USChange

__all__ = [
    "Airline",
    "ForecastingData",
    "HierarchicalSalesToydata",
    "Longley",
    "Lynx",
    "Macroeconomic",
    "PBS",
    "ShampooSales",
    "Solar",
    "USChange",
    "M5Dataset",
]
