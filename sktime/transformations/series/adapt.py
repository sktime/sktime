#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adaptor for applying Scikit-learn-like transformers to time series."""

__author__ = ["mloning"]
__all__ = ["TabularToSeriesAdaptor"]

from sklearn.base import clone

from sktime.transformations.base import BaseTransformer


class TabularToSeriesAdaptor(BaseTransformer):
    """Adapt scikit-learn-like  transformations to time series setting.

    This is useful for applying scikit-learn :term:`tabular` transformations
    to :term:`series <Time series>`, but only works with transformations that
    do not require multiple :term:`instances <instance>` for fitting.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like transformer to fit and apply to series.
        This is used as a "blueprint" and not fitted or otherwise mutated.

    Attributes
    ----------
    transformer_ : Estimator
        Transformer that is fitted to data, clone of transformer.
    fit_in_transform: bool, optional, default=False
        whether transformer_ should be fitted in transform (True), or in fit (False)
        WARNING: if this is set to False, when applied to Panel or Hierarchical data,
            the resulting transformer will identify individual series in test set
            with series indices in training set, on which instances were fit
            in particular, transform will not work if number of instances
                and indices of instances in transform are different from those in fit
        WARNING: if this is set to True,
            then series in the test set will be transformed as batch by fit-predict
            this may cause information leakage in a forecasting setting
                (but not in a time series classification/regression/clustering setting)

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
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "transform-returns-same-time-index": True,
        "fit-in-transform": False,
    }

    def __init__(self, transformer, fit_in_transform=False):
        self.transformer = transformer
        self.transformer_ = clone(self.transformer)
        self.fit_in_transform = fit_in_transform

        super(TabularToSeriesAdaptor, self).__init__()

        if hasattr(transformer, "inverse_transform"):
            self.set_tags(**{"capability:inverse_transform": True})

        if fit_in_transform:
            self.set_tags(**{"fit-in-transform": True})

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

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
        if not self.fit_in_transform:
            self.transformer_.fit(X)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

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
        if self.fit_in_transform:
            Xt = self.transformer_.fit(X).transform(X)
        else:
            Xt = self.transformer_.transform(X)
        return Xt

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        core logic

        Parameters
        ----------
        X : 2D np.ndarray
            Data to be inverse transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        if self.fit_in_transform:
            Xt = self.transformer_.fit(X).inverse_transform(X)
        else:
            Xt = self.transformer_.inverse_transform(X)
        return Xt

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.preprocessing import StandardScaler

        params1 = {"transformer": StandardScaler()}
        params2 = {"transformer": StandardScaler(), "fit_in_transform": True}

        return [params1, params2]
