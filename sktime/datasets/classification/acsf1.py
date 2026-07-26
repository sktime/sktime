"""ACSF1 dataset."""

from sktime.datasets._single_problem_loaders import load_acsf1
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class ACSF1(_ClassificationDatasetFromLoader):
    """ACSF1 time series classification problem.

    Example of a univariate classification dataset with long idle periods.

    Examples
    --------
    >>> from sktime.datasets.classification import ACSF1
    >>> X, y = ACSF1().load("X", "y")

    Notes
    -----
    Dimensionality:     univariate
    Series length:      1460
    Train cases:        100
    Test cases:         100
    Number of classes:  10

    The dataset contains the power consumption of typical appliances.
    The recordings are characterized by long idle periods and some high bursts
    of energy consumption when the appliance is active.
    The classes correspond to 10 categories of home appliances;
    mobile phones (via chargers), coffee machines, computer stations
    (including monitor), fridges and freezers, Hi-Fi systems (CD players),
    lamp (CFL), laptops (via chargers), microwave ovens, printers, and
    televisions (LCD or LED).

    Dataset details: http://www.timeseriesclassification.com/description.php?Dataset=ACSF1
    """

    _tags = {
        "name": "acsf1",
        "n_splits": 1,
        "is_univariate": True,
        "n_instances": 200,
        "n_instances_train": 100,
        "n_instances_test": 100,
        "n_classes": 10,
    }

    loader_func = load_acsf1
