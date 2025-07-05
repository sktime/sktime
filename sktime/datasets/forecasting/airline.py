"""Airline dataset."""

from sktime.datasets._single_problem_loaders import load_airline
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["Airline"]


class Airline(_ForecastingDatasetFromLoader):
    """Load the airline univariate time series dataset [1].

    Examples
    --------
    >>> from sktime.datasets.forecasting import Airline
    >>> dataset = Airline()
    >>> y = dataset.load("y")

    Notes
    -----
    The classic Box & Jenkins airline data. Monthly totals of international
    airline passengers, 1949 to 1960.

    Dimensionality:     univariate
    Series length:      144
    Frequency:          Monthly
    Number of cases:    1

    This data shows an increasing trend, non-constant (increasing) variance
    and periodic, seasonal patterns.

    References
    ----------
    .. [1] Box, G. E. P., Jenkins, G. M. and Reinsel, G. C. (1976) Time Series
          Analysis, Forecasting and Control. Third Edition. Holden-Day.
          Series G.
    """

    _tags = {
        "name": "airline",
        "is_univariate": True,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": False,
        "n_instances": 144,
        "n_timepoints": 144,
        "frequency": "M",
        "n_dimensions": 1,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = load_airline
