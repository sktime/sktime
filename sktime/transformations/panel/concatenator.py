# -*- coding: utf-8 -*-
from sktime.transformations.base import _PanelToTabularTransformer
import pandas as pd
import numpy as np

__author__ = ["Viktor Kazakov"]
__all__ = ["Concatenator"]


class Concatenator(_PanelToTabularTransformer):
    """
    Concatenates
    """

    def __init__(self):
        super(Concatenator, self).__init__()

    def fit_transform(self, X, y=None):
        """
        Parameters
        ----------
        X : list of pandas dataframes
        """
        if type(X) != list:
            # Only for passing the sktime checks. `X` must be a list.
            X = [X]
        if type(X) == pd.core.frame.DataFrame:
            return pd.concat(X, axis=1)
        if type(X) == np.ndarray:
            return np.concatenate(tuple(X), axis=1)

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : list of pandas dataframes
        """
        return self.fit_transform(X=X, y=y)
