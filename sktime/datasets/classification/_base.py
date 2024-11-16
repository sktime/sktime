# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for classification datasets."""

__author__ = ["fkiraly"]


from sktime.datasets.base import BaseDataset, _DatasetFromLoaderMixin

__all__ = [
    "BaseClassificationDataset",
    "_ClassificationDatasetFromLoader",
]


class BaseClassificationDataset(BaseDataset):
    """Base class for classification datasets."""

    _tags = {
        "object_type": "classification_dataset",
        # Estimator type
        "is_univariate": True,
        "n_panels": 1,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_equal_length": True,
        "is_equal_index": False,
        "is_empty": False,
        "has_nans": False,
        "n_instances": None,
        "n_instances_train": None,
        "n_instances_test": None,
        "n_classes": 2,
    }

    def __init__(self, return_mtype="pd-multiindex"):
        self.return_mtype = return_mtype
        super().__init__()

    def load(self, *args):
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
        # Validate args
        if any([_x in args for _x in ["X_test" "y_test"]]):
            if not self.get_tag("n_instances_test", 0):
                raise ValueError("Test data split not available for this dataset. ")
        if any([_x in args for _x in ["X_train", "y_train"]]):
            if not self.get_tag("n_instances_train", 0):
                raise ValueError("Train data split not available for this dataset. ")

        return self._load(*args)


class _ClassificationDatasetFromLoader(
    _DatasetFromLoaderMixin, BaseClassificationDataset
):
    """Classification dataset object, wrapping an sktime loader function."""
