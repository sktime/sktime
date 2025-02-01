"""US Macroeconomic dataset for multivariate time series forecasting."""

from sktime.datasets._single_problem_loaders import load_macroeconomic
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["Macroeconomic"]


class Macroeconomic(_ForecastingDatasetFromLoader):
    """Load the US Macroeconomic dataset for multivariate time series forecasting.

    Examples
    --------
    >>> from sktime.datasets.forecasting import Macroeconomic
    >>> y = Macroeconomic().load("y")

    Notes
    -----
    This dataset contains US Macroeconomic Data from 1959Q1 to 2009Q3.

    Dimensionality:     multivariate, 14
    Series length:      203
    Frequency:          Quarterly
    Number of cases:    1

    This data is kindly wrapped via ``statsmodels.datasets.macrodata``.

    References
    ----------
    .. [1] Wrapped via statsmodels:
          https://www.statsmodels.org/dev/datasets/generated/macrodata.html
    .. [2] Data Source: FRED, Federal Reserve Economic Data, Federal Reserve
          Bank of St. Louis; http://research.stlouisfed.org/fred2/;
          accessed December 15, 2009.
    .. [3] Data Source: Bureau of Labor Statistics, U.S. Department of Labor;
          http://www.bls.gov/data/; accessed December 15, 2009.
    """

    _tags = {
        "name": "macroeconomic",
        "python_dependencies": ["statsmodels"],
        "n_splits": 0,
        "is_univariate": False,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": False,
        "n_instances": 203,
        "n_timepoints": 203,
        "n_timepoints_train": 0,
        "n_timepoints_test": 0,
        "frequency": "Q",
        "n_dimensions": 14,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = load_macroeconomic

    def __init__(self):
        super().__init__()
