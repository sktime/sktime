"""USChange dataset for forecasting growth rates of consumption and income."""

import functools

from sktime.datasets._single_problem_loaders import load_uschange
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["USChange"]


class USChange(_ForecastingDatasetFromLoader):
    """Load USChange dataset for forecasting growth rates of consumption and income.

    Parameters
    ----------
    y_name : str, optional (default="Consumption")
        Name of the target variable (y).


    Examples
    --------
    >>> from sktime.datasets.forecasting import USChange
    >>> y, X = USChange().load("y", "X")

    Notes
    -----
    This dataset contains percentage changes in quarterly personal consumption
    expenditure, personal disposable income, production, savings, and the unemployment
    rate for the US from 1960 to 2016.

    Dimensionality:     multivariate
    Columns:            ['Consumption', 'Income', 'Production',
                         'Savings', 'Unemployment']
    Series length:      188
    Frequency:          Quarterly
    Number of cases:    1

    This data exhibits an increasing trend, non-constant (increasing) variance,
    and periodic, seasonal patterns.

    References
    ----------
    .. [1] Data for "Forecasting: Principles and Practice" (2nd Edition).
    """

    _tags = {
        "name": "uschange",
        "n_splits": 0,  # No splits available
        "is_univariate": False,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": True,
        "n_instances": 187,
        "n_timepoints": 187,
        "frequency": "Q",
        "n_dimensions": 5,  # 5 explanatory variables
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = functools.partial(
        load_uschange,
        y_name=["Consumption", "Income", "Production", "Savings", "Unemployment"],
    )

    def __init__(self):
        super().__init__()
