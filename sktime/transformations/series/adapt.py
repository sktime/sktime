#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["TabularToSeriesAdaptor"]

import pandas as pd
from sklearn.base import clone
from sklearn.utils.metaestimators import if_delegate_has_method

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


def _from_series_to_2d_numpy(x):
    x = x.to_numpy()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _from_2d_numpy_to_series(x, index=None):
    assert x.ndim < 3
    if x.ndim == 2 and x.shape[1] > 1:
        x = pd.DataFrame(x)
    else:
        x = pd.Series(x.ravel())
    if index is not None:
        x.index = index
    return x


class TabularToSeriesAdaptor(_SeriesToSeriesTransformer):
    """Adaptor for scikit-learn-like tabular transformations to series
    setting.

    This is useful for applying scikit-learn transformations to series,
    but only works with transformations that do not require multiple
    instances for fitting.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like transformer to fit and apply to series

    Example
    ----------
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = TabularToSeriesAdaptor(MinMaxScaler())
    >>> y_hat = transformer.fit_transform(y)
    """

    _required_parameters = ["transformer"]
    _tags = {"transform-returns-same-time-index": True}

    def __init__(self, transformer):
        self.transformer = transformer
        self.transformer_ = None
        super(TabularToSeriesAdaptor, self).__init__()

    def fit(self, Z, X=None):
        """Fit.

        Parameters
        ----------
        Z : TimeSeries
        X : TimeSeries

        Returns
        -------
        self
        """
        Z = check_series(Z)
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(_from_series_to_2d_numpy(Z))
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Transform data.
        Returns a transformed version of y.

        Parameters
        ----------
        y : pd.Series

        Returns
        -------
        yt : pd.Series
            Transformed time series.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        Zt = self.transformer_.transform(_from_series_to_2d_numpy(Z))
        return _from_2d_numpy_to_series(Zt, index=Z.index)

    @if_delegate_has_method(delegate="transformer")
    def inverse_transform(self, Z, X=None):
        """Inverse transform data.

        Parameters
        ----------
        Z : TimeSeries

        Returns
        -------
        Zt : TimeSeries
            Inverse-transformed time series.
        """
        self.check_is_fitted()
        Z = check_series(Z)
        Zt = self.transformer_.inverse_transform(_from_series_to_2d_numpy(Z))
        return _from_2d_numpy_to_series(Zt, index=Z.index)
