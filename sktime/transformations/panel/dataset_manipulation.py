# -*- coding: utf-8 -*-
from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.base import _PanelToTabularTransformer
import pandas as pd
import numpy as np

__author__ = ["Viktor Kazakov"]
__all__ = ["Selector", "Concatenator"]

"""Basic dataset manipulations"""


class Selector(_PanelToPanelTransformer):
    """Select columns from Pandas Dataframe.

    Parameters
    ----------
    columns: integer
    """

    _tags = {"fit-in-transform": True}

    def __init__(self, columns, return_dataframe=True):
        self.columns = columns
        self.return_dataframe = return_dataframe
        super(Selector, self).__init__()

    def transform(self, X, y=None):
        """
        Public transform method.

        Parameters
        ----------
        X : pd DataFrame or np.array
        """
        self.check_is_fitted()
        if type(X) == pd.core.frame.DataFrame:
            if self.return_dataframe:
                return X.iloc[:, self.columns].to_frame()
            else:
                return X.iloc[:, self.columns]
        if type(X) == np.ndarray:
            return X[:, self.columns]


class Concatenator(_PanelToTabularTransformer):
    """Concatenate pandas series or numpy arrays."""

    _tags = {
        "fit-in-transform": True,
    }

    def __init__(self):
        super(Concatenator, self).__init__()

    def transform(self, X, y=None):
        """
        Fit and transform.

        Parameters
        ----------
        X : list of pandas dataframes
        """
        self.check_is_fitted()
        if type(X) != list:
            # Only for passing the sktime checks. `X` must be a list.
            return X

        if type(X) == pd.core.frame.DataFrame:
            return pd.concat(X, axis=1)
        if type(X) == np.ndarray:
            return np.concatenate(tuple(X), axis=1)
