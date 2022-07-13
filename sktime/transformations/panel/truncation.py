# -*- coding: utf-8 -*-
"""Truncation transformer - truncate unequal length panels to lower/upper bounds."""
import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__all__ = ["TruncationTransformer"]
__author__ = ["abostrom"]


class TruncationTransformer(BaseTransformer):
    """Truncate unequal length panels to lower/upper bounds.

    Truncates all series in panel between lower/upper range bounds.

    Parameters
    ----------
    lower : int, optional (default=None) bottom range of the values to
                truncate can also be used to truncate to a specific length
                if None, will find the shortest sequence and use instead.
    upper : int, optional (default=None) upper range, only required when
                paired with lower.
                This is used to calculate the range between. exclusive.
                if None, will truncate from 0 to the lower bound.
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "capability:unequal_length:removes": True,
        # is transform result always guaranteed to be equal length (and series)?
    }

    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper
        self.min_length = lower
        super(TruncationTransformer, self).__init__()

    @staticmethod
    def _get_min_length(X):
        def get_length(input):
            return min(map(lambda series: len(series), input))

        return min(map(get_length, X))

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        if self.lower is None:
            n_instances, _ = X.shape
            arr = [X.iloc[i, :].values for i in range(n_instances)]
            self.lower_ = self._get_min_length(arr)
        else:
            self.lower_ = self.lower

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series
            transformed version of X
        """
        n_instances, _ = X.shape

        arr = [X.iloc[i, :].values for i in range(n_instances)]

        min_length = self._get_min_length(arr)

        if min_length < self.lower_:
            raise ValueError(
                "Error: min_length of series \
                    is less than the one found when fit or set."
            )

        # depending on inputs either find the shortest truncation.
        # or use the bounds.
        if self.upper is None:
            idxs = np.arange(self.lower_)
        else:
            idxs = np.arange(self.lower_, self.upper)

        truncate = [pd.Series([series.iloc[idxs] for series in out]) for out in arr]

        Xt = pd.DataFrame(truncate)
        Xt.columns = X.columns

        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = {"lower": 5}
        return params
