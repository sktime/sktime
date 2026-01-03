"""UCR UEA Datasets."""

__author__ = ["jgyasu"]
__all__ = ["UCRUEADataset"]

from inspect import signature

from sktime.datasets import load_UCR_UEA_dataset
from sktime.datasets.classification._base import BaseClassificationDataset
from sktime.datasets.classification.ucr_uea_archive._tags import DATASET_TAGS
from sktime.datatypes import convert_to


class UCRUEADataset(BaseClassificationDataset):
    """UCR-UEA time series classification dataset loader.

    Generic dataset loader for problems from the UCR UEA repository.
    Provides access to both univariate and multivariate classification
    datasets.

    Parameters
    ----------
    name: str
        Name of the dataset to load.

    Examples
    --------
    >>> from sktime.datasets import UCRUEADataset # doctest: +SKIP
    >>> dataset = UCRUEADataset(name="Beef") # doctest: +SKIP
    >>> X, y = dataset.load() # doctest: +SKIP
    >>> X_train, y_train = dataset.load("X_train", "y_train") # doctest: +SKIP
    >>> all_names = UCRUEADataset.list_all() # doctest: +SKIP

    Notes
    -----
    Dimensionality:     univariate or multivariate (depends on dataset)
    Series length:      varies across datasets
    Train cases:        varies across datasets
    Test cases:         varies across datasets
    Number of classes:  varies across datasets

    The UCR/UEA repository is the primary benchmark archive for time series
    classification. It contains a wide range of datasets from domains such as
    medicine, motion capture, sensor recordings, speech recognition, and image
    outline analysis. Many datasets are univariate with equal-length series,
    but the archive also includes multivariate and unequal-length problems.

    Dataset details: https://timeseriesclassification.com/dataset.php
    """

    def __init__(self, name, return_mtype="pd-multiindex"):
        super().__init__(return_mtype=return_mtype)
        self.name = name
        self.loader_func = load_UCR_UEA_dataset

        self.set_tags(name=self.name)
        self.set_tags(**DATASET_TAGS[self.name])

    def _encode_args(self, code):
        """Decide kwargs for the loader function."""
        kwargs = {}

        if code in ["X", "y"]:
            split = None
        elif code in ["X_train", "y_train"]:
            split = "TRAIN"
        elif code in ["X_test", "y_test"]:
            split = "TEST"
        else:
            split = None

        loader_func_params = signature(self.loader_func).parameters
        init_signature_params = signature(self.__init__).parameters
        init_param_values = {k: getattr(self, k) for k in init_signature_params.keys()}

        if (
            "test" in code.lower() or "train" in code.lower()
        ) and "split" not in loader_func_params:
            raise ValueError(
                "This dataset loader does not have a train/test split. "
                "Load the full dataset instead."
            )

        if "split" in loader_func_params:
            kwargs["split"] = split

        for init_param_name, init_param_value in init_param_values.items():
            if init_param_name in loader_func_params:
                kwargs[init_param_name] = init_param_value

        return kwargs

    def _load(self, *args):
        """Load the dataset.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            "X": full panel data set of instances to classify
            "y": full set of class labels
            "X_train": training instances only, for fixed single split
            "y_train": training labels only, for fixed single split
            "X_test": test instances only, for fixed single split
            "y_test": test labels only, for fixed single split

        Returns
        -------
        dataset, if args is empty or length one
            data container corresponding to string in args (see above)
        tuple, of same length as args, if args is length 2 or longer
            data containers corresponding to strings in args, in same order
        """
        if len(args) == 0:
            args = ("X", "y")

        cache = {}

        if "X" in args or "y" in args:
            X, y = self._load_dataset(**self._encode_args("X"))
            cache["X"] = X
            cache["y"] = y
        if "X_train" in args or "y_train" in args:
            X, y = self._load_dataset(**self._encode_args("X_train"))
            cache["X_train"] = X
            cache["y_train"] = y
        if "X_test" in args or "y_test" in args:
            X, y = self._load_dataset(**self._encode_args("X_test"))
            cache["X_test"] = X
            cache["y_test"] = y
        if "cv" in args:
            cv = self._load_simple_train_test_cv_split()
            cache["cv"] = cv

        res = [cache[key] for key in args]
        return res[0] if len(res) == 1 else tuple(res)

    def _load_dataset(self, **kwargs):
        """Call loader function with self.name included automatically."""
        if "name" in signature(self.loader_func).parameters and "name" not in kwargs:
            kwargs["name"] = self.name
        X, y = self.loader_func(**kwargs)
        X = convert_to(X, "pd-multiindex")
        return (X, y)

    def list_all():
        """List all the datasets loadable via `UCRUEADataset` class.

        Returns
        -------
        datasets: list of str
            list of all the loadable datasets via `UCRUEADataset`
        """
        datasets = list(DATASET_TAGS.keys())
        return datasets

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter."""
        params_list = [
            {
                "name": "Beef",
            },
            {
                "name": "BeetleFly",
            },
        ]

        return params_list
