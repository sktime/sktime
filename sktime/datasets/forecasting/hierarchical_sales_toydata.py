"""Hierarchical sales toy data for hierarchical forecasting."""

from sktime.datasets._hierarchical_demo import load_hierarchical_sales_toydata
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["HierarchicalSalesToydata"]


class HierarchicalSalesToydata(_ForecastingDatasetFromLoader):
    """Return hierarchical sales toy data to demonstrate hierarchical forecasting.

    Return data covers 5 years of monthly sales data for 2 product lines,
    and 4 product groups.

    Hierarchical structure:

    - Product line 1: Food preparation
        - Product group 1A: Hobs
        - Product group 1B: Ovens
    - Product line 2: Food preservation
        - Product group 2A: Fridges
        - Product group 2B: Freezers

    Examples
    --------
    >>> from sktime.datasets.forecasting import HierarchicalSalesToydata
    >>> dataset = HierarchicalSalesToydata()
    >>> y = dataset.load("y")
    """

    _tags = {
        "name": "hierarchical_sales_toydata",
        "is_univariate": True,
        "is_one_series": False,
        "is_one_panel": False,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": False,
        "n_instances": 12 * 5 * 4,
        "n_timepoints": 12 * 5 * 4,
        "frequency": "M",
        "n_dimensions": 1,
        "n_panels": 4,
        "n_hierarchy_levels": 2,
    }

    loader_func = load_hierarchical_sales_toydata
