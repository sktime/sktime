# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X

__all__ = ["TruncationTransformer"]
__author__ = ["Aaron Bostrom"]


class TruncationTransformer(_PanelToPanelTransformer):
    """TruncationTransformer docstring

    Parameters
    ----------
    lower   : int, optional (default=None) bottom range of the values to
                truncate can also be used to truncate to a specific length

                if None, will find the shortest sequence and use instead.

    upper   : int, optional (default=None) upper range, only required when
                paired with lower.
                This is used to calculate the range between. exclusive.

                if None, will truncate from 0 to the lower bound.
    """

    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper
        self.min_length = lower
        super(TruncationTransformer, self).__init__()

    @staticmethod
    def get_min_length(X):
        def get_length(input):
            return min(map(lambda series: len(series), input))

        return min(map(get_length, X))

    def fit(self, X, y=None):
        """
        Fit transformer.

        Parameters
        ----------
        X : pandas DataFrame of shape [n_samples, n_features]
            Input data
        y : pandas Series, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        self : an instance of self.
        """
        X = check_X(X, coerce_to_pandas=True)

        if self.lower is None:
            n_instances, _ = X.shape
            arr = [X.iloc[i, :].values for i in range(n_instances)]
            self.lower_ = self.get_min_length(arr)
        else:
            self.lower_ = self.lower

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """
        Transform X.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_columns]
            Nested dataframe with time-series in cells.

        Returns
        -------
        Xt : pandas DataFrame
        """
        self.check_is_fitted()
        X = check_X(X, coerce_to_pandas=True)

        n_instances, _ = X.shape

        arr = [X.iloc[i, :].values for i in range(n_instances)]

        min_length = self.get_min_length(arr)

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

        return pd.DataFrame(truncate)
