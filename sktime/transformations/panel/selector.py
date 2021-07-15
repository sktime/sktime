# -*- coding: utf-8 -*-
from sktime.transformations.base import _PanelToPanelTransformer

__author__ = ["Viktor Kazakov"]
__all__ = ["Selector"]


class Selector(_PanelToPanelTransformer):
    """Selects columns from Pandas Dataframe

    Parameters
    ----------

    columns: integer
    """

    _tags = {"fit-in-transform": True}

    def __init__(self, columns):
        self.columns = columns
        super(Selector, self).__init__()

    def fit_transform(self, X, y=None):
        """
        Parameters
        ----------

        X : pd DataFrame
        """

        return X.iloc[:, self.columns].to_frame()

    def transform(self, X, y=None):
        return self.fit_transform(X=X, y=y)
