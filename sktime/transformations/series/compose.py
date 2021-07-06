#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Martin Walter", "Svea Meyer"]
__all__ = ["OptionalPassthrough", "ColumnComposition"]

import pandas as pd
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series

from sklearn.base import clone
from sklearn.utils.metaestimators import if_delegate_has_method


class OptionalPassthrough(_SeriesToSeriesTransformer):
    """A transformer to tune the implicit hyperparameter whether or not to use a
    particular transformer inside a pipeline (e.g. TranformedTargetForecaster)
    or not. This is achived by having the additional hyperparameter
    "passthrough" which can be added to a grid then (see example).

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series
    passthrough : bool
        This arg decides whether to apply the given transformer or to just
        passthrough the data (identity transformation)

    Example
    ----------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.series.compose import OptionalPassthrough
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.model_selection import (
    ...     ForecastingGridSearchCV,
    ...     SlidingWindowSplitter)
    >>> from sklearn.preprocessing import StandardScaler
    >>> # create pipeline
    >>> pipe = TransformedTargetForecaster(steps=[
    ...     ("deseasonalizer", OptionalPassthrough(Deseasonalizer())),
    ...     ("scaler", OptionalPassthrough(TabularToSeriesAdaptor(StandardScaler()))),
    ...     ("forecaster", NaiveForecaster())])
    >>> # putting it all together in a grid search
    >>> cv = SlidingWindowSplitter(
    ...     initial_window=60,
    ...     window_length=24,
    ...     start_with_window=True,
    ...     step_length=48)
    >>> param_grid = {
    ...     "deseasonalizer__passthrough" : [True, False],
    ...     "scaler__transformer__transformer__with_mean": [True, False],
    ...     "scaler__passthrough" : [True, False],
    ...     "forecaster__strategy": ["drift", "mean", "last"]}
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=pipe,
    ...     param_grid=param_grid,
    ...     cv=cv,
    ...     n_jobs=-1)
    >>> gscv_fitted = gscv.fit(load_airline())
    """

    _required_parameters = ["transformer"]
    _tags = {
        "univariate-only": True,
        "fit-in-transform": True,
    }

    def __init__(self, transformer, passthrough=False):
        self.transformer = transformer
        self.transformer_ = None
        self.passthrough = passthrough
        self._is_fitted = False
        super(OptionalPassthrough, self).__init__()

    def fit(self, Z, X=None):
        if not self.passthrough:
            self.transformer_ = clone(self.transformer)
            self.transformer_.fit(Z, X)
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        if not self.passthrough:
            z = self.transformer_.transform(z, X)
        return z

    @if_delegate_has_method(delegate="transformer")
    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        if not self.passthrough:
            z = self.transformer_.inverse_transform(z, X=None)
        return z


class ColumnComposition(_SeriesToSeriesTransformer):
    """
    Applies a transformer for univariate series
    columnwise to multivariate series.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series
    columns : list of str
            Names of columns that are supposed to be transformed
    """

    _required_parameters = ["transformer"]

    def __init__(self, transformer, columns="all"):
        self.transformer = transformer
        self.columns = columns
        self.transformers_ = None
        super(ColumnComposition, self).__init__()

    def fit(self, Z, X=None):
        """
        Iterates over columns (series) and applies
        the fit function of the transformer.

        Parameters
        ----------
        Z : pd.Series, pd.DataFrame

        Returns
        -------
        self : an instance of self

        """
        z = check_series(Z, allow_numpy=False)
        self._is_fitted = False

        # univariate case
        if isinstance(z, pd.Series):
            self.transformer.fit(z, X)
            self._is_fitted = True
            return self
        # multivariate case
        else:
            if self.columns == "all":
                self.columns = z.columns

            # make sure z contains all columns that the user wants to transform
            Z_wanted_keys = set(self.columns)
            Z_new_keys = set(z.columns)
            difference = Z_wanted_keys.difference(Z_new_keys)
            if len(difference) != 0:
                raise ValueError("Missing columns" + str(difference) + "in Z.")

            self.transformers_ = {}
            for colname in self.columns:
                transformer = clone(self.transformer)
                self.transformers_[colname] = transformer
                self.transformers_[colname].fit(z[colname], X)
            self._is_fitted = True
            return self

    def transform(self, Z, X=None):
        """Transform data.
        Returns a transformed version of Z by iterating over specified
        columns and applying the univariate series transformer to them.

        Parameters
        ----------
        Z : pd.Series, pd.DataFrame

        Returns
        -------
        Z : pd.Series, pd.DataFrame
            Transformed time series(es).
        """
        self.check_is_fitted()
        z = check_series(Z)

        # univariate case
        if isinstance(z, pd.Series):
            z = self.transformer.transform(z, X)
            return z

        # multivariate case
        else:
            # make copy of z
            z = z.copy()
            # make sure z contains all columns that the user wants to transform
            Z_wanted_keys = set(self.columns)
            Z_new_keys = set(z.columns)
            difference = Z_wanted_keys.difference(Z_new_keys)
            if len(difference) != 0:
                raise ValueError("Missing columns" + str(difference) + "in Z.")
            for colname in self.columns:
                # self.columns : columns that are supposed to be transformed
                self.transformers_[colname].check_is_fitted()
                z[colname] = self.transformers_[colname].transform(z[colname], X)
            return z

    def inverse_transform(self, Z, X=None):
        """
        Inverse-transform data.

        Returns an inverse-transformed version of Z by iterating over specified
        columns and applying the univariate series transformer to them.
        Only works if self.transformer has an inverse-transform method.

        Parameters
        ----------
        Z : pd.Series, pd.DataFrame

        Returns
        -------
        Z : pd.Series, pd.DataFrame
            Inverse-transformed time series(es).
        """
        self.check_is_fitted()
        z = check_series(Z)

        # univariate case
        if isinstance(z, pd.Series):
            z = self.transformer.inverse_transform(z, X)
            return z

        # multivariate case
        else:
            # make copy of z
            z = z.copy()

            # make sure z contains all columns that the user wants to transform
            Z_wanted_keys = set(self.columns)
            Z_new_keys = set(z.columns)
            difference = Z_wanted_keys.difference(Z_new_keys)
            if len(difference) != 0:
                raise ValueError("Missing columns" + str(difference) + "in Z.")
            for colname in self.columns:
                # self.columns : columns that are supposed to be transformed
                z[colname] = self.transformers_[colname].inverse_transform(
                    z[colname], X
                )
            return z

    def update(self, Z, X=None, update_params=True):
        """
        Update the parameters of the estimator with new data
        by iterating over specified columns.
        Only works if self.transformer has an update method.

        Parameters
        ----------
        Z : pd.Series
            New time series
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        z = check_series(Z)

        # univariate case
        if isinstance(z, pd.Series):
            self.transformer.update(z, X)
            return self

        # multivariate case
        else:
            # make copy of z
            z = z.copy()

            # make sure z contains all columns that the user wants to transform
            Z_wanted_keys = set(self.columns)
            Z_new_keys = set(z.columns)
            difference = Z_wanted_keys.difference(Z_new_keys)
            if len(difference) != 0:
                raise ValueError("Missing columns" + str(difference) + "in Z.")
            for colname in self.columns:
                # self.columns : columns that are supposed to be transformed
                z[colname] = self.transformers_[colname].update(z[colname], X)
            return self
