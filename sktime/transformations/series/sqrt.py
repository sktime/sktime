#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Class to iteratively apply differences to a time series."""
__author__ = ["Ryan Kuhns"]
__all__ = ["SqrtTransformer"]

import numpy as np

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class SqrtTransformer(_SeriesToSeriesTransformer):
    """Apply square root transformation to a timeseries.

    Example
    -------
    >>> from sktime.transformations.series.sqrt import SqrtTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SqrtTransformer()
    >>> y_transform = transformer.fit_transform(y)
    """

    _tags = {
        "fit-in-transform": False,
        "transform-returns-same-time-index": True,
        "univariate-only": False,
    }

    def __init__(self, lags=1, remove_missing=True):
        super(SqrtTransformer, self).__init__()

    def _fit(self, Z, X=None):
        """Logic used by fit method on `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the transformation on.

        Returns
        -------
        self
        """

        return self

    def _transform(self, Z, X=None):
        """Logic used by `transform` to apply transformation to `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            The timeseries to be transformed.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            Transformed timeseries.
        """
        Zt = np.sqrt(Z)
        return Zt

    def _inverse_transform(self, Z, X=None):
        """Logic used by `inverse_transform` to reverse transformation on  `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply reverse the transformation on.

        Returns
        -------
        Z_inv : pd.Series or pd.DataFrame
            The reconstructed timeseries after the transformation has been reversed.
        """
        Z_inv = np.square(Z)
        return Z_inv

    def fit(self, Z, X=None):
        """Fit the transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the transformation on.

        Returns
        -------
        self
        """
        Z = check_series(Z)

        self._fit(Z, X=X)

        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Return transformed version of input series `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to apply the transformation on.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            Transformed version of input series `Z`.
        """
        self.check_is_fitted()
        Z = check_series(Z)

        Zt = self._transform(Z, X=X)

        return Zt

    def inverse_transform(self, Z, X=None):
        """Reverse transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.Series or pd.DataFrame
            A time series to reverse the transformation on.

        Returns
        -------
        Z_inv : pd.Series or pd.DataFrame
            The reconstructed timeseries after the transformation has been reversed.
        """
        self.check_is_fitted()
        Z = check_series(Z)

        Z_inv = self._inverse_transform(Z, X=X)

        return Z_inv
