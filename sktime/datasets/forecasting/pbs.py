"""PBS dataset."""

from sktime.datasets._single_problem_loaders import load_PBS_dataset
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["PBS"]


class PBS(_ForecastingDatasetFromLoader):
    """Load the Pharmaceutical Benefit Scheme (PBS) univariate time series dataset [1]_.

    Examples
    --------
    >>> from sktime.datasets.forecasting import PBS
    >>> dataset = PBS()
    >>> y = dataset.load("y")

    Notes
    -----
    The Pharmaceutical Benefits Scheme (PBS) is the Australian government drugs
    subsidy scheme. Data comprises the number of scripts sold each month for
    immune sera and immunoglobulin products in Australia.

    Dimensionality:     univariate
    Series length:      204
    Frequency:          Monthly
    Number of cases:    1

    The time series is intermittent, i.e. it contains small counts, with many months
    registering no sales at all, and only small numbers of items sold in other months.

    References
    ----------
    .. [1] Data for "Forecasting: Principles and Practice" (3rd Edition).
    """

    _tags = {
        "name": "pbs",
        "is_univariate": True,
        "is_one_series": True,
        "is_one_panel": True,
        "is_empty": False,
        "has_exogenous": False,
        "n_instances": 1,
        "n_timepoints": 204,
        "frequency": "M",
        "n_dimensions": 1,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = load_PBS_dataset
