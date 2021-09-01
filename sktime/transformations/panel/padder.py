# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X

__all__ = ["PaddingTransformer"]
__author__ = ["Aaron Bostrom"]


class PaddingTransformer(_PanelToPanelTransformer):
    """PaddingTransformer docstring

    Pads the input dataset to either a optional fixed length
    (longer than the longest series).
    Or finds the max length series across all series and dimensions and
    pads to that with zeroes.

    Parameters
    ----------
    pad_length  : int, optional (default=None) length to pad the series too.

                if None, will find the longest sequence and use instead.
    """

    def __init__(self, pad_length=None, fill_value=0):
        self.pad_length = pad_length
        self.fill_value = fill_value
        super(PaddingTransformer, self).__init__()

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

        if self.pad_length is None:
            n_instances, _ = X.shape
            arr = [X.iloc[i, :].values for i in range(n_instances)]
            self.pad_length_ = _get_max_length(arr)
        else:
            self.pad_length_ = self.pad_length

        self._is_fitted = True
        return self

    def _create_pad(self, series):
        out = np.full(self.pad_length_, self.fill_value, np.float)
        out[: len(series)] = series.iloc[: len(series)]
        return out

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

        n_instances, n_dims = X.shape

        arr = [X.iloc[i, :].values for i in range(n_instances)]

        max_length = _get_max_length(arr)

        if max_length > self.pad_length_:
            raise ValueError(
                "Error: max_length of series \
                    is greater than the one found when fit or set."
            )

        pad = [pd.Series([self._create_pad(series) for series in out]) for out in arr]

        return pd.DataFrame(pad)


def _get_max_length(X):
    def get_length(input):
        return max(map(lambda series: len(series), input))

    return max(map(get_length, X))
