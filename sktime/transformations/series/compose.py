#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Martin Walter", "Svea Meyer"]
__all__ = ["OptionalPassthrough", "ColumnwiseTransformer"]

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
        "univariate-only": False,
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
        z = check_series(Z, enforce_univariate=False)
        if not self.passthrough:
            z = self.transformer_.transform(z, X)
        return z

    @if_delegate_has_method(delegate="transformer")
    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=False)
        if not self.passthrough:
            z = self.transformer_.inverse_transform(z, X=None)
        return z


class ColumnwiseTransformer(_SeriesToSeriesTransformer):
    """
    Applies a transformer for univariate series
    columnwise to multivariate series.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series
    columns : list of str
            Names of columns that are supposed to be transformed

    Attributes
    ----------
    transformers_ : dict of {str : transformer}
        Maps columns to transformers

    Example
    -------
    >>> from sktime.datasets import base
    >>> from sktime.transformations.series.detrend import Detrender
    >>> from sktime.transformations.series.compose import ColumnwiseTransformer

    >>> y, X = load_longley()
    >>> transformer = ColumnwiseTransformer(Detrender())
    >>> yt = transformer.fit_transform(X)
    """

    _required_parameters = ["transformer"]

    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns

        # check that columns are None or list of strings
        if columns is not None:
            if not isinstance(columns, list) and all(
                isinstance(s, str) for s in columns
            ):
                raise ValueError("Columns need to be a list of strings or None.")

        self.columns_ = columns
        self.transformers_ = None
        super(ColumnwiseTransformer, self).__init__()

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
            z = z.to_frame()

        if self.columns_ is None:
            self.columns_ = z.columns

        # make sure z contains all columns that the user wants to transform
        self._check_columns(z)

        self.transformers_ = {}
        for colname in self.columns_:
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
        is_series = False

        # univariate case
        if isinstance(z, pd.Series):
            is_series = True
            name_of_z = z.name
            z = z.to_frame()

        # make copy of z
        z = z.copy()

        # make sure z contains all columns that the user wants to transform
        self._check_columns(z)
        for colname in self.columns_:
            # self.columns_ : columns that are supposed to be transformed
            z[colname] = self.transformers_[colname].transform(z[colname], X)

        # make z a series again in univariate case
        if is_series:
            z = self._revert_to_series(name_of_z, z)
        return z

    @if_delegate_has_method(delegate="transformer")
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

        is_series = False
        # univariate case
        if isinstance(z, pd.Series):
            name_of_z = z.name
            z = z.to_frame()
            is_series = True

        # make copy of z
        z = z.copy()

        # make sure z contains all columns that the user wants to transform
        self._check_columns(z)

        for colname in self.columns_:
            # self.columns_ : columns that are supposed to be transformed
            z[colname] = self.transformers_[colname].inverse_transform(z[colname], X)
        # make z a series again in univariate case
        if is_series:
            z = self._revert_to_series(name_of_z, z)
        return z

    @if_delegate_has_method(delegate="transformer")
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
            z = z.to_frame()

        # make sure z contains all columns that the user wants to transform
        self._check_columns(z)
        for colname in self.columns_:
            # self.columns_ : columns that are supposed to be transformed
            self.transformers_[colname].update(z[colname], X)
        return self

    def _check_columns(self, z):
        # make sure z contains all columns that the user wants to transform
        z_wanted_keys = set(self.columns_)
        z_new_keys = set(z.columns)
        difference = z_wanted_keys.difference(z_new_keys)
        if len(difference) != 0:
            raise ValueError("Missing columns" + str(difference) + "in Z.")

    def _revert_to_series(self, name_of_z, z):
        if name_of_z is not None:
            z = z[name_of_z]
        else:
            z = z[0]
            z.name = None
        return z
