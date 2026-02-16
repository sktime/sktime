"""ItalyPowerDemand dataset."""

from sktime.datasets._single_problem_loaders import load_italy_power_demand
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class ItalyPowerDemand(_ClassificationDatasetFromLoader):
    """ItalyPowerDemand time series classification problem.

    Example of a univariate problem with equal-length series.

    Examples
    --------
    >>> from sktime.datasets.classification import ItalyPowerDemand
    >>> X, y = ItalyPowerDemand().load("X", "y")

    Notes
    -----
    Dimensionality:     univariate
    Series length:      24
    Train cases:        67
    Test cases:         1029
    Number of classes:  2

    The data was derived from twelve monthly electrical power demand time series from
    Italy and was first used in the paper "Intelligent Icons: Integrating Lite-Weight
    Data Mining and Visualization into GUI Operating Systems". The classification task
    is to distinguish days from October to March (inclusive) from April to September.

    Dataset details:
    http://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
    """

    _tags = {
        "name": "italy_power_demand",
        "n_splits": 1,
        "is_univariate": True,
        "n_instances": 1096,
        "n_instances_train": 67,
        "n_instances_test": 1029,
        "n_classes": 2,
    }

    loader_func = load_italy_power_demand
