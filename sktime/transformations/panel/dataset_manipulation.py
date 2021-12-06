# -*- coding: utf-8 -*-
"""Network Pipeline transformers.

Selection of simple transformers that can be used with
`sktime.forecasting.compose._networkpipeline.NetworkPipelineForecaster`
"""

import numpy as np
import pandas as pd

from sktime.datatypes._convert import convert_to
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested_adp
from sktime.transformations.base import _PanelToPanelTransformer

__author__ = ["Viktor Kazakov"]
__all__ = ["Selector", "SeriesUnion", "Converter"]


class Selector(_PanelToPanelTransformer):
    """Select columns from Pandas Dataframe.

    Parameters
    ----------
    columns : integer
    convert_to_dataframe : Bool
    """

    _tags = {"fit-in-transform": True}

    def __init__(self, columns, convert_to_dataframe=True):
        self.columns = columns
        self.convert_to_dataframe = convert_to_dataframe
        super(Selector, self).__init__()

    def transform(self, X, y=None):
        """
        Public transform method.

        Parameters
        ----------
        X : pd DataFrame
        """
        self.check_is_fitted()
        if type(X) == pd.core.frame.DataFrame:
            if self.convert_to_dataframe:
                return X.iloc[:, self.columns].to_frame()
            else:
                return X.iloc[:, self.columns]
        if type(X) == np.ndarray:
            return X[:, self.columns]

    def fit_transform(self, X, y=None):
        """Use for passing unit tests."""
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X, y=None):
        """Use for passing unit tests."""
        self._is_fitted = True
        return self


class SeriesUnion(_PanelToPanelTransformer):
    """Concatenate pandas series or numpy arrays."""

    _tags = {
        "fit-in-transform": True,
    }

    def __init__(self):
        super(SeriesUnion, self).__init__()

    def transform(self, X, y=None):
        """
        Public transform method.

        Parameters
        ----------
        X : list
            pandas dataframes or numpy arrays
        """
        self.check_is_fitted()
        if type(X) != list:
            # Only for passing the sktime checks. `X` must be a list.
            return X
        else:
            for i in range(len(X)):
                if i == 0:
                    continue
                else:
                    if not isinstance(X[i], type(X[0])):
                        raise ValueError("All elements of X must be of the same type.")
            if (type(X[0]) == pd.core.frame.DataFrame) or (
                type(X[0]) == pd.core.series.Series
            ):
                return pd.concat(X, axis=1)
            if type(X[0]) == np.ndarray:
                return np.concatenate(tuple(X), axis=1)

    def fit_transform(self, X, y=None):
        """Use for passing unit tests."""
        self.fit(X, y)
        return self.transform(X, y)

    def fit(self, X, y=None):
        """Use for passing unit tests."""
        self._is_fitted = True
        return self


class Converter(_PanelToPanelTransformer):
    """Wraps sktime.datatypes._convert.

    Transformer interface that can be used within pipelines
    The default values for to_type and as_scitype
    are given for passing the unit tests.
    """

    _tags = {
        "fit-in-transform": True,
    }

    def __init__(self):
        super(Converter, self).__init__()

    def transform(self, obj, to_type="pd.DataFrame", as_scitype="Series", store=None):
        """Public transform method.

        Transformer calls sktime.datatypes._convert.

        Parameters
        ----------
        obj : any type supported by sktime.datatypes._convert.convert_to
        to_type : str
        as_scitype : str

        Returns
        -------
            resulting type from `sktime.datatypes._convert.convert_to`
        """
        self.check_is_fitted()
        # for passing unit tests.
        # If 3d numpy array as passed return pandas DataFrame
        if (type(obj) is np.ndarray) and len(obj.shape) == 3:
            return from_3d_numpy_to_nested_adp(obj)
        if type(obj) is pd.DataFrame:
            # for passing unit tests.
            return obj
        return convert_to(obj, to_type, as_scitype, store)

    def fit_transform(
        self, obj, to_type="pd.DataFrame", as_scitype="Series", store=None
    ):
        """Use for passing unit tests."""
        self.fit(obj, to_type, as_scitype, store)
        return self.transform(obj, to_type, as_scitype, store)

    def fit(self, obj, to_type="pd.DataFrame", as_scitype="Series", store=None):
        """Use for passing unit tests."""
        self._is_fitted = True
        return self
