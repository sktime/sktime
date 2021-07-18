# -*- coding: utf-8 -*-
from sktime.transformations.base import _PanelToTabularTransformer
import pandas as pd
import numpy as np

__author__ = ["Viktor Kazakov"]
__all__ = ["Concatenator"]


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
