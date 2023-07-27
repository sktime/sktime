# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for classification datasets."""

__author__ = ["fkiraly"]

from inspect import isfunction

from sktime.datasets.base import BaseDataset


class BaseClassificationDataset(BaseDataset):
    """Base class for classification datasets."""

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
        pass


def _coerce_to_list_of_str(obj):
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


class _ClassificationDatasetFromLoader(BaseClassificationDataset):
    """Classification dataset object, wrapping an sktime loader function."""

    loader_func = None

    def _encode_args(self, code):
        kwargs = {}
        if code in ["X", "y"]:
            split = None
        elif code in ["X_train", "y_train"]:
            split = "TRAIN"
        elif code in ["X_test", "y_test"]:
            split = "TRAIN"
        else:
            split = None
        kwargs = {"split": split, "return_type": self.return_mtype}
        return kwargs

    def load(self, *args):
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
        # calls class variable loader_func, if available, or dynamic (object) variable
        # we need to call type since we store func as a class attribute
        if hasattr(type(self), "loader_func") and isfunction(type(self).loader_func):
            loader = type(self).loader_func
        else:
            loader = self.loader_func

        if len(args) == 0:
            args = ("X", "y")

        cache = {}
        if "X" in args or "y" in args:
            X, y = loader(**self._encode_args("X"))
            cache["X"] = X
            cache["y"] = y
        if "X_train" in args or "y_train" in args:
            X, y = loader(**self._encode_args("X_train"))
            cache["X_train"] = X
            cache["y_train"] = y
        if "X_test" in args or "y_test" in args:
            X, y = loader(**self._encode_args("X_test"))
            cache["X_test"] = X
            cache["y_test"] = y

        res = [cache[key] for key in args]
        res = tuple(res)

        return res
