# -*- coding: utf-8 -*-
"""Base class template for transformers."""

__author__ = ["mloning"]
__all__ = [
    "BaseTransformer",
    "_SeriesToPrimitivesTransformer",
    "_SeriesToSeriesTransformer",
    "_PanelToTabularTransformer",
    "_PanelToPanelTransformer",
]

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
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit-in-transform": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
    }

    def __init__(self):
        super(BaseTransformer, self).__init__()

    def fit(self, Z, X=None):
        """Fit transformer to X and y.

        By default, fit is empty. Fittable transformations overwrite fit method.

        Parameters
        ----------
        Z : TimeSeries
            Data to be transformed
        X : TimeSeries, optional (default=None)
            Extra data

        Returns
        -------
        self : a fitted instance of the estimator
        """
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Transform data. Returns a transformed version of Z."""
        raise NotImplementedError("abstract method")

    def fit_transform(self, Z, X=None):
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
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        if X is None:
            # Fit method of arity 1 (unsupervised transformation)
            return self.fit(Z).transform(Z)
        else:
            # Fit method of arity 2 (supervised transformation)
            return self.fit(Z, X).transform(Z)

    # def inverse_transform(self, Z, X=None):
    #     raise NotImplementedError("abstract method")
    #
    # def update(self, Z, X=None, update_params=False):
    #     raise NotImplementedError("abstract method")


class _SeriesToPrimitivesTransformer(BaseTransformer):
    """Transformer base class for series to primitive(s) transforms."""

    def transform(self, Z: Series, X=None) -> Primitives:
        """Transform data. Returns a transformed version of Z."""
        raise NotImplementedError("abstract method")


class _SeriesToSeriesTransformer(BaseTransformer):
    """Transformer base class for series to series transforms."""

    def transform(self, Z: Series, X=None) -> Series:
        """Transform data. Returns a transformed version of Z."""
        raise NotImplementedError("abstract method")


class _PanelToTabularTransformer(BaseTransformer):
    """Transformer base class for panel to tabular transforms."""

    def transform(self, X: Panel, y=None) -> Tabular:
        """Transform data. Returns a transformed version of X."""
        raise NotImplementedError("abstract method")


class _PanelToPanelTransformer(BaseTransformer):
    """Transformer base class for panel to panel transforms."""

    def transform(self, X: Panel, y=None) -> Panel:
        """Transform data. Returns a transformed version of X."""
        raise NotImplementedError("abstract method")
