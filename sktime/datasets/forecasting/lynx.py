"""Lynx dataset."""

from sktime.datasets._single_problem_loaders import load_lynx
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["Lynx"]


class Lynx(_ForecastingDatasetFromLoader):
    """Load the Lynx dataset for univariate time series forecasting.

    Examples
    --------
    >>> from sktime.datasets.forecasting import Lynx
    >>> y = Lynx().load("y")

    Notes
    -----
    This dataset contains the annual number of lynx trappings in Canada from 1821 to
    1934.
    It records the number of lynx skins collected over several years by the Hudson's Bay
    Company.
    The data was obtained from Brockwell & Davis (1991) and has been analyzed by
    Campbell & Walker (1977).

    Dimensionality:     univariate
    Series length:      114
    Frequency:          Yearly
    Number of cases:    1

    This data exhibits aperiodic, cyclical patterns rather than periodic, seasonal
    patterns.

    References
    ----------
    .. [1] Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988). The New S
    Language. Wadsworth & Brooks/Cole.

    .. [2] Campbell, M. J. and Walker, A. M. (1977). A Survey of statistical
    work on the Mackenzie River series of annual Canadian lynx trappings for the years
    1821-1934 and a new analysis. Journal of the Royal Statistical Society series A,
    140, 411-431.
    """

    _tags = {
        "name": "lynx",
        "n_splits": 0,
        "is_univariate": True,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": False,
        "n_instances": 114,
        "n_timepoints": 114,
        "n_timepoints_train": None,
        "n_timepoints_test": None,
        "frequency": "Y",
        "n_dimensions": 1,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = load_lynx

    def __init__(self):
        super().__init__()
