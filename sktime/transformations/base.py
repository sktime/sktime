# -*- coding: utf-8 -*-
"""
Base class template for transformers.

    class name: BaseTransformer

Covers all types of transformers.
Type and behaviour of transformer is determined by the following tags:
    "scitype:transform-output" tag with values "Primivites", or "Series"
        this determines type of output of transform
        if "Primitives", output is pd.DataFrame with as many rows as X has instances
            i-th instance of X is transformed into i-th row of output
        if "Series", output is a Series or Panel, with as many instances as X
            i-th instance of X is transformed into i-th instance of output
        Series are treated as one-instance-Panels
            if Series is input, output is a 1-row pd.DataFrame or a Series
    "scitype:instancewise" tag which is boolean
        if True, fit/transform is statistically independent by instance

Scitype defining methods:
    fitting         - fit(self, X, y=None)
    transform       - transform(self, X, y=None)
    fit&transform   - fit_transform(self, X, y=None)
    updating        - update(self, X, y=None)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - is_fitted (property)
    fitted state inspection - check_is_fitted()

copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["Markus Löning"]
__all__ = [
    "BaseTransformer",
    "_SeriesToPrimitivesTransformer",
    "_SeriesToSeriesTransformer",
    "_PanelToTabularTransformer",
    "_PanelToPanelTransformer",
]

import warnings

from typing import Union

import numpy as np
import pandas as pd

from sktime.base import BaseEstimator

# single/multiple primitives
Primitive = Union[np.integer, int, np.float, float, str]
Primitives = np.ndarray

# tabular/cross-sectional data
Tabular = Union[pd.DataFrame, np.ndarray]  # 2d arrays

# univariate/multivariate series
UnivariateSeries = Union[pd.Series, np.ndarray]
MultivariateSeries = Union[pd.DataFrame, np.ndarray]
Series = Union[UnivariateSeries, MultivariateSeries]

# panel/longitudinal/series-as-features data
Panel = Union[pd.DataFrame, np.ndarray]  # 3d or nested array


class BaseTransformer(BaseEstimator):
    """Transformer base class."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Panel or Series
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce-index-type": None,  # index type that needs to be enforced in X/y
    }

    def __init__(self):
        super(BaseTransformer, self).__init__()

    def fit(self, X, y=None, Z=None):
        """
        Fit transformer to X and y.

        By default, fit is empty. Fittable transformations overwrite fit method.

        Parameters
        ----------
        X : TimeSeries
            Data to be transformed
        y : TimeSeries, optional (default=None)
            Extra data

        Returns
        -------
        self : a fitted instance of the estimator
        """
        X = _handle_alias(X, Z)

        self._is_fitted = True

        self.fit(X=X, y=y)

        return self

    def transform(self, X, y=None, Z=None):
        """Transform data. Returns a transformed version of X."""
        X = _handle_alias(X, Z)

        return self._transform(X=X, y=y)

    def fit_transform(self, X, y=None, Z=None):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        Z : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        X : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be transformed.

        Returns
        -------
        Zt : pd.DataFrame, pd.Series or np.ndarray
            Transformed data.
        """
        X = _handle_alias(X, Z)
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y, Z).transform(X, y, Z)
#
    # def inverse_transform(self, Z, X=None):
    #     raise NotImplementedError("abstract method")
    #
    # def update(self, Z, X=None, update_params=False):
    #     raise NotImplementedError("abstract method")

    def _fit(self, X, y=None):
        """
        Fit transformer to X and y.

        core logic

        Parameters
        ----------
        X : TimeSeries
            Data to be transformed
        y : TimeSeries, optional (default=None)
            Extra data

        Returns
        -------
        self : a fitted instance of the estimator
        """
        # default fit is "no fitting happens"
        return self

    def _transform(self, X, y=None):
        """Transform data. Returns a transformed version of X."""
        raise NotImplementedError("abstract method")


def _handle_alias(X, Z):
    """Handle Z as an alias for X, return X/Z.

    Parameters
    ----------
    X: any object
    Z: any object

    Returns
    -------
    X if Z is None, Z if X is None

    Raises
    ------
    ValueError both X and Z are not None
    """
    if Z is None:
        return X
    elif X is None:
        warnings.warn(
            "argument Z will be deprecated in transformers, sktime version 0.9.0"
        )
        return Z
    else:
        raise ValueError("X and Z are aliases, at most one of them should be passed")


class _SeriesToPrimitivesTransformer(BaseTransformer):
    """Transformer base class for series to primitive(s) transforms"""

    def transform(self, Z: Series, X=None) -> Primitives:
        raise NotImplementedError("abstract method")


class _SeriesToSeriesTransformer(BaseTransformer):
    """Transformer base class for series to series transforms"""

    def transform(self, Z: Series, X=None) -> Series:
        raise NotImplementedError("abstract method")


class _PanelToTabularTransformer(BaseTransformer):
    """Transformer base class for panel to tabular transforms"""

    def transform(self, X: Panel, y=None) -> Tabular:
        raise NotImplementedError("abstract method")


class _PanelToPanelTransformer(BaseTransformer):
    """Transformer base class for panel to panel transforms"""

    def transform(self, X: Panel, y=None) -> Panel:
        raise NotImplementedError("abstract method")
