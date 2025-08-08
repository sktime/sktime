# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for classification datasets."""

__author__ = ["felipeangelimvieira"]


from sktime.datasets.base import BaseDataset, _DatasetFromLoaderMixin

__all__ = [
    "BaseForecastingDataset",
    "_ForecastingDatasetFromLoader",
]


class BaseForecastingDataset(BaseDataset):
    """Base class for Forecasting datasets.

    Tags
    ----

    is_univariate: bool, default=True
        Whether the dataset is univariate. In the case of Forecasting dataset,
        this refers to the dimensionality of the `y` dataframe.
    is_equally_spaced: bool, default=True
        Whether the all obserations in the dataset are equally spaced.
    has_nans: bool, default=False
        True if the dataset contains NaN values, False otherwise.
    has_exogenous: bool, default=False
        True if the dataset contains exogenous variables, False otherwise.
    n_instances: int, default=None
        Number of instances in the dataset. Should be equal to y.shape[0].
    n_instances_train: int, default=None
        Number of instances in the training set. None if the dataset does not
        have a train/test split. Should be equal to y_train.shape[0].
    n_instances_test: int, default=None
        Number of instances in the test set. None if the dataset does not
        have a train/test split. Should be equal to y_test.shape[0].
    n_timepoints: int, default=None
        Number of timepoints in the dataset, per series. If the dataset is composed
        of series of different lengths, this should be equal the max length
        seen in the dataset.
    n_timepoints_train: int, default=None
        Number of timepoints in the training set, per series. If the dataset is composed
        of series of different lengths, this should be equal the max length seen
        in the training set.
    n_timepoints_test: int, default=None
        Number of timepoints in the test set, per series. If the dataset is composed
        of series of different lengths, this should be equal the max length seen
        in the test set.
    frequency: str, default=None
        Frequency of the time series in the dataset. Can be an integer if the
        frequency is not related to a time unit.
    n_dimensions: int, default=1
        Number of dimensions in the dataset. This is the number of columns in
        the `y` dataframe.
    n_panels: int, default=1
        Number of panels in the dataset. This is the number of unique time series
        in the dataset.
    n_hierarchy_levels: int, default=0
        Number of levels in the hierarchy of the dataset. This is the number of
        index levels in the `y` dataframe, excluding the time index.


    """

    _tags = {
        "object_type": ["dataset_forecasting", "dataset"],
        "task_type": ["forecaster"],
        # Estimator type
        "is_univariate": True,
        "is_equally_spaced": True,
        "has_nans": False,
        "has_exogenous": False,
        "n_instances": None,
        "n_instances_train": 0,
        "n_instances_test": 0,
        "n_timepoints": None,
        "n_timepoints_train": None,
        "n_timepoints_test": None,
        "frequency": "M",
        "n_dimensions": 1,
        "is_one_panel": True,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
        "is_one_series": True,  # Number of levels  in the hierarchy (equivalent to
        #  number of index levels excluding the time index)
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
