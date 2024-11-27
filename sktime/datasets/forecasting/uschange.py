"""USChange dataset for forecasting growth rates of consumption and income."""

from sktime.datasets._single_problem_loaders import load_uschange
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["USChange"]


class USChange(_ForecastingDatasetFromLoader):
    """Load USChange dataset for forecasting growth rates of consumption and income.

    Parameters
    ----------
    y_name : str, optional (default="Consumption")
        Name of the target variable (y).

    Returns
    -------
    y : pd.Series
        Selected target column, default is "Consumption".
    X : pd.DataFrame
        Explanatory variables.

    Examples
    --------
    >>> from sktime.datasets.forecasting import USChange
    >>> y, X = USChange().load()

    Notes
    -----
    This dataset contains percentage changes in quarterly personal consumption
    expenditure, personal disposable income, production, savings, and the unemployment
    rate for the US from 1960 to 2016.

    Dimensionality:     multivariate
    Columns:            ['Quarter', 'Consumption', 'Income', 'Production',
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
        "is_univariate": False,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": True,
        "n_instances": 188,
        "n_instances_train": 0,  # Can vary if a default split is defined
        "n_instances_test": 0,
        "frequency": "Q",
        "n_dimensions": 5,  # 5 explanatory variables
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = load_uschange

    def __init__(self, y_name="Consumption"):
        self.y_name = y_name
        super().__init__()
