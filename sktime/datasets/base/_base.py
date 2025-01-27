# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Base class template for data sets.

    class name: BaseDataset

Scitype defining methods:
    loading dataset              - load()
    loading object from dataset  - load(*args)

Inspection methods:
    hyper-parameter inspection   - get_params()
"""

__author__ = ["fkiraly", "felipeangelimvieira"]

__all__ = ["BaseDataset", "_DatasetFromLoaderMixin"]

import shutil
from inspect import isfunction, signature
from pathlib import Path

from sktime.base import BaseObject
from sktime.utils.dependencies import _check_estimator_deps


class BaseDataset(BaseObject):
    """Base class for datasets."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "object_type": "dataset",  # type of object
        "name": None,  # The dataset unique name
        "python_dependencies": None,  # python dependencies required to load the dataset
        "python_version": None,  # python version required to load the dataset
        "n_splits": 0,  # Number of cross-validation splits, if any.
    }

    def __init__(self):
        super().__init__()
        _check_estimator_deps(self)

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
        if len(args) == 0:
            args = ("X", "y")
        self._check_args(*args)

        return self._load(*args)

    def _load_simple_train_test_cv_split(self):
        """
        Return the cv split for datasets with a single split.

        Returns
        -------
        generator
            Generator that yields X_train, y_train, X_test, y_test

        Raises
        ------
        ValueError
            If the dataset has more than one split.
        """
        n_splits = self.get_tag("n_splits")

        if n_splits == 1:
            X_train, y_train = self.load("X_train", "y_train")
            X_test, y_test = self.load("X_test", "y_test")
            # Return X_train, y_train, X_test, y_test in a generator
            yield X_train, y_train, X_test, y_test

        raise ValueError("This method is only for datasets with a single split.")

    def _check_args(self, *args):
        for arg in args:
            if arg not in self.keys():
                raise InvalidSetError(arg, self.keys())

    def keys(self):
        """
        Return a list of available sets.

        Returns
        -------
        list of str
            List of available sets.
        """
        sets = ["X", "y"]
        n_splits = self.get_tag("n_splits")
        if n_splits == 1:
            sets.extend(["X_train", "y_train", "X_test", "y_test"])
        elif n_splits > 1:
            sets.append("cv")
        return sets

    def cache_files_directory(self):
        """
        Get the directory where cache files are stored.

        Returns
        -------
        Path
            Directory where cache files are stored
        """
        dataset_name = self.get_tag("name")
        return Path(__file__).parent.parent / Path("data") / dataset_name

    def cleanup_cache_files(self):
        """Cleanup cache files from the cache directory."""
        cache_directory = self.cache_files_directory()
        if cache_directory.exists():
            shutil.rmtree(cache_directory)

    def __getitem__(self, key):
        return self.load(key)


class InvalidSetError(Exception):
    """Exception raised for invalid set names."""

    def __init__(self, set_name, valid_set_names):
        self.set_name = set_name
        self.valid_set_names = valid_set_names

    def __str__(self):
        return (
            f"Invalid set name: {self.set_name}. "
            f"Valid set names are: {self.valid_set_names}."
        )


class _DatasetFromLoaderMixin:
    loader_func = None

    def _encode_args(self, code):
        kwargs = {}
        if code in ["X", "y"]:
            split = None
        elif code in ["X_train", "y_train"]:
            split = "TRAIN"
        elif code in ["X_test", "y_test"]:
            split = "TEST"
        else:
            split = None

        # Check if loader_func has split and return_type parameters
        # else set kwargs = {}
        loader = self.get_loader_func()
        loader_func_params = signature(loader).parameters
        init_signature_params = signature(self.__init__).parameters
        init_param_values = {k: getattr(self, k) for k in init_signature_params.keys()}

        if (
            "test" in code.lower() or "train" in code.lower()
        ) and "split" not in loader_func_params:
            raise ValueError(
                "This dataset loader does not have a train/test split"
                + "Load the full dataset instead."
            )

        if "split" in loader_func_params:
            kwargs["split"] = split

        for init_param_name, init_param_value in init_param_values.items():
            if init_param_name in loader_func_params:
                kwargs[init_param_name] = init_param_value

        return kwargs

    def get_loader_func(self):
        # calls class variable loader_func, if available, or dynamic (object) variable
        # we need to call type since we store func as a class attribute
        if hasattr(type(self), "loader_func") and isfunction(type(self).loader_func):
            loader = type(self).loader_func
        else:
            loader = self.loader_func
        return loader

    def _load(self, *args):
        """Load the dataset.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            "X": full panel data set of instnaces to classify
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

        # Returns a single element if there is only one element in the list
        if len(res) == 1:
            res = res[0]
        # Else, returns a tuple
        else:
            res = tuple(res)

        return res

    def _load_dataset(self, **kwargs):
        """
        Call loader function and return dataset dataframes.

        This method is intended to be overridden by child classes if the order of `X`
        and `y` in the loader output is different from the default order (`X` first, `y`
        second).
        """
        loader_func = self.get_loader_func()
        return loader_func(**kwargs)
