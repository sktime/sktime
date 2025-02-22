"""Longley dataset for forecasting with exogenous variables."""

from sktime.datasets._single_problem_loaders import load_longley
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["Longley"]


class Longley(_ForecastingDatasetFromLoader):
    """Load the Longley dataset for forecasting with exogenous variables.

    Parameters
    ----------
    y_name: str, optional (default="TOTEMP")
        Name of target variable (y)

    Examples
    --------
    >>> from sktime.datasets.forecasting import Longley
    >>> y, X = Longley().load("y", "X")

    Notes
    -----
    This mulitvariate time series dataset contains various US macroeconomic
    variables from 1947 to 1962 that are known to be highly collinear.

    Dimensionality:     multivariate, 6
    Series length:      16
    Frequency:          Yearly
    Number of cases:    1

    Variable description:

    TOTEMP - Total employment
    GNPDEFL - Gross national product deflator
    GNP - Gross national product
    UNEMP - Number of unemployed
    ARMED - Size of armed forces
    POP - Population

    References
    ----------
    .. [1] Longley, J.W. (1967) "An Appraisal of Least Squares Programs for the
        Electronic Computer from the Point of View of the User."  Journal of
        the American Statistical Association.  62.319, 819-41.
        (https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/Longley.dat)
    """

    _tags = {
        "name": "longley",
        "is_univariate": True,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": True,
        "n_instances": 16,
        "n_timepoints": 16,
        "frequency": "Y",
        "n_dimensions": 1,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = load_longley

    def __init__(self, y_name: str = "TOTEMP"):
        self.y_name = y_name
        super().__init__()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameter settings for the dataset."""
        return [
            {"y_name": "TOTEMP"},
            {"y_name": "GNPDEFL"},
            {"y_name": "GNP"},
            {"y_name": "UNEMP"},
            {"y_name": "ARMED"},
            {"y_name": "POP"},
        ]
