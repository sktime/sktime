"""Shampoo Sales dataset."""

from sktime.datasets._single_problem_loaders import load_shampoo_sales
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["ShampooSales"]


class ShampooSales(_ForecastingDatasetFromLoader):
    """Load the Shampoo Sales dataset for univariate time series forecasting.

    Examples
    --------
    >>> from sktime.datasets.forecasting import ShampooSales
    >>> y = ShampooSales().load("y")

    Notes
    -----
    This dataset describes the monthly number of shampoo sales over a 3-year period.
    The units are in sales count.

    Dimensionality:     univariate
    Series length:      36
    Frequency:          Monthly
    Number of cases:    1

    References
    ----------
    .. [1] Makridakis, Wheelwright and Hyndman (1998) Forecasting: methods
       and applications, John Wiley & Sons: New York. Chapter 3.
    """

    _tags = {
        "name": "shampoo_sales",
        "n_splits": 0,
        "is_univariate": True,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": False,
        "n_instances": 36,
        "n_timepoints": 36,
        "frequency": "M",
        "n_dimensions": 1,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = load_shampoo_sales

    def __init__(self):
        super().__init__()
