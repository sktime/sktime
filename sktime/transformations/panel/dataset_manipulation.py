# -*- coding: utf-8 -*-
from sktime.transformations.base import _PanelToPanelTransformer
import pandas as pd
import numpy as np
from sktime.datatypes._panel._convert import from_3d_numpy_to_nested_adp
from sktime.datatypes._convert import convert_to

"""
Provide basic transformers to use together with
`sktime.forecasting.compose._networkpipeline.NetworkPipelineForecaster`

"""

__author__ = ["Viktor Kazakov"]
__all__ = ["Selector", "Concatenator", "Converter"]


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
        X : pd DataFrame
        """
        self.check_is_fitted()
        if type(X) == pd.core.frame.DataFrame:
            if self.return_dataframe:
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


class Concatenator(_PanelToPanelTransformer):
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
        else:
            # further checks need to be run to ensure
            # all elements of the list are the same
            if type(X[0]) == pd.core.frame.DataFrame:
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
        """
        Public transform method.

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
