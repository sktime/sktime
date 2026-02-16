# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for classification datasets."""

__author__ = ["felipeangelimvieira"]


from sktime.datasets.base import BaseDataset, _DatasetFromLoaderMixin

__all__ = [
    "BaseClassificationDataset",
    "_ClassificationDatasetFromLoader",
]


class BaseClassificationDataset(BaseDataset):
    """Base class for classification datasets.

    Tags
    ----

    is_univariate: bool, default=True
        Whether the dataset is univariate. In the case of classification dataset,
        this refers to the dimensionality of X dataframe, i.e., how many series are
        related to a class label.
    n_instances: int, default=None
        Number of instances in the dataset.
    n_instances_train: int, default=None
        Number of instances in the training set.
    n_instances_test: int, default=None
        Number of instances in the test set.
    n_classes: int, default=2
        Number of classes in the dataset.

    """

    _tags = {
        "object_type": ["dataset_classification", "dataset"],
        "task_type": ["classifier"],
        # Estimator type
        "is_univariate": True,
        "n_instances": None,
        "n_instances_train": None,
        "n_instances_test": None,
        "n_classes": 2,
        "reserved_params": ["return_mtype"],
    }

    def __init__(self, return_mtype="pd-multiindex"):
        self.return_mtype = return_mtype
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


class _ClassificationDatasetFromLoader(
    _DatasetFromLoaderMixin, BaseClassificationDataset
):
    """Classification dataset object, wrapping an sktime loader function."""
