#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adaptor for applying Scikit-learn-like transformers to time series."""

__author__ = ["mloning"]
__all__ = ["TabularToSeriesAdaptor"]

import pandas as pd
from sklearn.base import clone
from sklearn.utils.metaestimators import if_delegate_has_method

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.series import check_series


class TabularToSeriesAdaptor(BaseTransformer):
    """Adapt scikit-learn-like  transformations to time series setting.

    This is useful for applying scikit-learn :term:`tabular` transformations
    to :term:`series <Time series>`, but only works with transformations that
    do not require multiple :term:`instances <instance>` for fitting.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like transformer to fit and apply to series.

    Attributes
    ----------
    transformer_ : Estimator
        Transformer fitted to data.

    Examples
    --------
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = TabularToSeriesAdaptor(MinMaxScaler())
    >>> y_hat = transformer.fit_transform(y)
    """

    _required_parameters = ["transformer"]

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "np.ndarray",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "univariate-only": True,
        "transform-returns-same-time-index": True,
        "fit-in-transform": False,
    }

    def __init__(self, transformer):
        self.transformer = transformer
        self.transformer_ = None
        super(TabularToSeriesAdaptor, self).__init__()

    def _fit(self, X, y=None):
        """
        Fit transformer to X and y.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(X)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : 2D np.ndarray
        transformed version of X
        """
        Xt = self.transformer_.transform(X)
        return Xt

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
