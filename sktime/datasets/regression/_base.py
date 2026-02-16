# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for classification datasets."""

__author__ = ["felipeangelimvieira"]


from sktime.datasets.base import BaseDataset, _DatasetFromLoaderMixin

__all__ = [
    "BaseRegressionDataset",
    "_RegressionDatasetFromLoader",
]


class BaseRegressionDataset(BaseDataset):
    """Base class for classification datasets.

    Tags
    ----
    is_univariate: bool, default=True
        Whether the dataset is univariate. In the case of regression dataset,
        this refers to the dimensionality of X dataframe, i.e., how many series are
        related to a class label.
    n_instances: int, default=None
        Number of instances in the dataset.
    n_instances_train: int, default=None
        Number of instances in the training set.
    n_instances_test: int, default=None
        Number of instances in the test set.
    """

    _tags = {
        "object_type": ["dataset_regression", "dataset"],
        "task_type": ["regressor"],
        # Estimator type
        "is_univariate": True,
        "n_instances": 215,
        "n_instances_train": 172,
        "n_instances_test": 43,
    }

    def __init__(self):
        super().__init__()

    def _load(self, *args):
        """Load the dataset.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            available/valid strings are provided by the concrete classes
            the expectation is that this docstring is replaced with the details

        Returns
        -------
        dataset, if args is empty or length one
            data container corresponding to string in args (see above)
        tuple, of same length as args, if args is length 2 or longer
            data containers corresponding to strings in args, in same order
        """
        raise NotImplementedError(
            "This method should be implemented by the child class."
        )


class _RegressionDatasetFromLoader(_DatasetFromLoaderMixin, BaseRegressionDataset):
    """Base class for classification datasets, when wrapping a loader func."""
