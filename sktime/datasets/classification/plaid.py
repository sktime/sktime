"""Plaid dataset."""

from sktime.datasets._single_problem_loaders import load_plaid
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class PLAID(_ClassificationDatasetFromLoader):
    """PLAID time series classification problem.

    Example of a univariate problem with unequal length series.

    Examples
    --------
    >>> from sktime.datasets.classification.plaid import PLAID
    >>> X, y = PLAID().load("X", "y")
    """

    _tags = {
        "name": "plaid",
        "n_splits": 1,
        "is_univariate": True,
        "n_instances": 1074,
        "n_instances_train": 537,
        "n_instances_test": 537,
        "n_classes": 11,
    }

    loader_func = load_plaid
