# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for classification datasets."""

__author__ = ["fkiraly"]


from sktime.datasets.base import BaseDataset, _DatasetFromLoaderMixin

__all__ = [
    "BaseForecastingDataset",
    "_ForecastingDatasetFromLoader",
]


class BaseForecastingDataset(BaseDataset):
    """Base class for classification datasets."""

    _tags = {
        "object_type": "forecasting_dataset",
        # Estimator type
        "is_univariate": True,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,
        "has_exogenous": False,
        "n_instances": None,
        "n_instances_train": None,
        "n_instances_test": None,
        "frequency": "M",
        "n_dimensions": 1,
        "n_panels": 1,
        "n_hierarchy_levels": 0,  # Number of levels  in the hierarchy (equivalent to
        #  number of index levels excluding the time index)
    }

    def __init__(self):
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
        if any([_x in args for _x in ["X_test", "y_test"]]):
            if not self.get_tag("n_instances_test", 0):
                raise ValueError("Test data split not available for this dataset. ")
        if any([_x in args for _x in ["X_train", "y_train"]]):
            if not self.get_tag("n_instances_train", 0):
                raise ValueError("Train data split not available for this dataset. ")

        return self._load(*args)

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


class _ForecastingDatasetFromLoader(_DatasetFromLoaderMixin, BaseForecastingDataset):
    def _load_dataset(self, **kwargs):
        loader_func = self.get_loader_func()
        output = loader_func(**kwargs)
        y, X = self._split_into_y_and_X(output)
        return X, y

    def _split_into_y_and_X(self, loader_output):
        """Split the output of the loader into X and y.

        Parameters
        ----------
        loader_output: any
            Output of the loader function.

        Returns
        -------
        tuple
            Tuple containing y and X.
        """
        if isinstance(loader_output, tuple):
            return loader_output

        y = loader_output
        X = None
        return (y, X)
