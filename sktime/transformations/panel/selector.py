# -*- coding: utf-8 -*-
from sktime.transformations.base import _PanelToPanelTransformer
import pandas as pd
import numpy as np

__author__ = ["Viktor Kazakov"]
__all__ = ["Selector"]


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
