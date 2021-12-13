#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Meta-transformers for building composite transformers."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.metaestimators import if_delegate_has_method

from sktime.forecasting.base import ForecastingHorizon
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.utils.validation.series import check_series

__author__ = ["aiwalter", "SveaMeyer13"]
__all__ = ["OptionalPassthrough", "ColumnwiseTransformer", "Featureizer"]


class OptionalPassthrough(_SeriesToSeriesTransformer):
    """Wrap an existing transformer to tune whether to include it in a pipeline.

    Allows tuning the implicit hyperparameter whether or not to use a
    particular transformer inside a pipeline (e.g. TranformedTargetForecaster)
    or not. This is achieved by the hyperparameter `passthrough`
    which can be added to a tuning grid then (see example).

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series.
    passthrough : bool, default=False
       Whether to apply the given transformer or to just
        passthrough the data (identity transformation). If, True the transformer
        is not applied and the OptionalPassthrough uses the identity
        transformation.

    Examples
    --------
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
        self.clone_tags(transformer)

    def fit(self, Z, X=None):
        """Fit the model.

        Parameters
        ----------
        Z : pd.Series
             Series to fit.
        X : pd.DataFrame, optional (default=None)
             Exogenous data used in transformation.

        Returns
        -------
        self
        """
        if not self.passthrough:
            self.transformer_ = clone(self.transformer)
            self.transformer_.fit(Z, X)
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Apply transformation.

        Parameters
        ----------
        Z : pd.Series
            Series to transform.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        z : pd.Series
            Transformed series.
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=False)
        if not self.passthrough:
            z = self.transformer_.transform(z, X)
        return z

    @if_delegate_has_method(delegate="transformer")
    def inverse_transform(self, Z, X=None):
        """Inverse transform data.

        Parameters
        ----------
        Z : pd.Series
            Series to transform.
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation.

        Returns
        -------
        z : pd.Series
            Inverse transformed data.
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=False)
        if not self.passthrough:
            z = self.transformer_.inverse_transform(z, X=X)
        return z


class ColumnwiseTransformer(_SeriesToSeriesTransformer):
    """Apply a transformer columnwise to multivariate series.

    Overview: input multivariate time series and the transformer passed
    in `transformer` parameter is applied to specified `columns`, each
    column is handled as a univariate series. The resulting transformed
    data has the same shape as input data.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series.
    columns : list of str or None
            Names of columns that are supposed to be transformed.
            If None, all columns are transformed.

    Attributes
    ----------
    transformers_ : dict of {str : transformer}
        Maps columns to transformers.
    columns_ : list of str
        Names of columns that are supposed to be transformed.

    See Also
    --------
    OptionalPassthrough

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.transformations.series.detrend import Detrender
    >>> from sktime.transformations.series.compose import ColumnwiseTransformer
    >>> _, X = load_longley()
    >>> transformer = ColumnwiseTransformer(Detrender())
    >>> Xt = transformer.fit_transform(X)
    """

    _required_parameters = ["transformer"]

    def __init__(self, transformer, columns=None):
        self.transformer = transformer
        self.columns = columns
        super(ColumnwiseTransformer, self).__init__()

    def fit(self, Z, X=None):
        """Fit data.

        Iterates over columns (series) and applies
        the fit function of the transformer.

        Parameters
        ----------
        Z : pd.Series, pd.DataFrame

        Returns
        -------
        self : an instance of self
        """
        self._is_fitted = False

        z = check_series(Z, allow_numpy=False)

        # cast to pd.DataFrame in univariate case
        if isinstance(z, pd.Series):
            z = z.to_frame()

        # check that columns are None or list of strings
        if self.columns is not None:
            if not isinstance(self.columns, list) and all(
                isinstance(s, str) for s in self.columns
            ):
                raise ValueError("Columns need to be a list of strings or None.")

        # set self.columns_ to columns that are going to be transformed
        # (all if self.columns is None)
        self.columns_ = self.columns
        if self.columns_ is None:
            self.columns_ = z.columns

        # make sure z contains all columns that the user wants to transform
        _check_columns(z, selected_columns=self.columns_)

        # fit by iterating over columns
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

        # handle univariate case
        z, is_series = _check_is_pdseries(z)

        # make copy of z
        z = z.copy()

        # make sure z contains all columns that the user wants to transform
        _check_columns(z, selected_columns=self.columns_)
        for colname in self.columns_:
            z[colname] = self.transformers_[colname].transform(z[colname], X)

        # make z a series again in univariate case
        if is_series:
            z = z.squeeze("columns")
        return z

    @if_delegate_has_method(delegate="transformer")
    def inverse_transform(self, Z, X=None):
        """Inverse transform data.

        Returns an inverse-transformed version of Z by iterating over specified
        columns and applying the univariate series transformer to them.
        Only works if `self.transformer` has an `inverse_transform` method.

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

        # handle univariate case
        z, is_series = _check_is_pdseries(z)

        # make copy of z
        z = z.copy()

        # make sure z contains all columns that the user wants to transform
        _check_columns(z, selected_columns=self.columns_)

        # iterate over columns that are supposed to be inverse_transformed
        for colname in self.columns_:
            z[colname] = self.transformers_[colname].inverse_transform(z[colname], X)

        # make z a series again in univariate case
        if is_series:
            z = z.squeeze("columns")
        return z

    @if_delegate_has_method(delegate="transformer")
    def update(self, Z, X=None, update_params=True):
        """Update parameters.

        Update the parameters of the estimator with new data
        by iterating over specified columns.
        Only works if `self.transformer` has an `update` method.

        Parameters
        ----------
        Z : pd.Series
            New time series.
        update_params : bool, optional, default=True

        Returns
        -------
        self : an instance of self
        """
        z = check_series(Z)

        # make z a pd.DataFrame in univariate case
        if isinstance(z, pd.Series):
            z = z.to_frame()

        # make sure z contains all columns that the user wants to transform
        _check_columns(z, selected_columns=self.columns_)
        for colname in self.columns_:
            self.transformers_[colname].update(z[colname], X)
        return self


def _check_columns(z, selected_columns):
    # make sure z contains all columns that the user wants to transform
    z_wanted_keys = set(selected_columns)
    z_new_keys = set(z.columns)
    difference = z_wanted_keys.difference(z_new_keys)
    if len(difference) != 0:
        raise ValueError("Missing columns" + str(difference) + "in Z.")


def _check_is_pdseries(z):
    # make z a pd.Dataframe in univariate case
    is_series = False
    if isinstance(z, pd.Series):
        z = z.to_frame()
        is_series = True
    return z, is_series


class Featureizer(_SeriesToSeriesTransformer):
    """Create new exogenous features based on a given transformer.

    Parameters
    ----------
    transformer: _SeriesToSeriesTransformer

    Attributes
    ----------
    transformer_ : _SeriesToSeriesTransformer
        Fitted transformer.
    """

    _tags = {
        "fit-in-transform": False,
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
        "univariate-only": False,
        "y_inner_mtype": ["pd.DataFrame"],
        "X_inner_mtype": ["pd.Series"],
        "scitype:X": "univariate",
        "scitype:y": "multivariate",
    }

    def __init__(self, transformer, fh, suffix=None):
        self.transformer = transformer
        self.fh = fh
        self.suffix = suffix
        super(Featureizer, self).__init__()

    def _fit(self, X, y=None):
        """
        Fit transformer to X and y.

        core logic

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _fit must support all types in it
            Data to fit transform to
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for tarnsformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        X = check_series(X, enforce_multivariate=True)
        y = check_series(y, enforce_univariate=True)
        # store y in self to use it in transform for outsample transformation
        self._y = y.copy()
        if not isinstance(self.transformer, _SeriesToSeriesTransformer):
            raise TypeError("Given transformer must be a _SeriesToSeriesTransformer")

        self.transformer_ = clone(self.transformer)
        # swap X, y
        self.transformer_.fit(y)
        # set suffix from transformer class name if None
        self.suffix_ = (
            self.suffix
            if self.suffix is not None
            else self.transformer.__class__.__name__.lower()
        )
        self._is_fitted = True
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        core logic

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        self.check_is_fitted()
        X = check_series(X, enforce_multivariate=True)
        y = check_series(y, enforce_univariate=True)
        Xt = X.copy()
        # check that fh is complete
        if not self.fh.is_complete:
            raise ValueError(
                "Given fh is not complete (has gaps). Please give a complete fh."
            )
        # check relative fh starts with 1 (no gap between cutoff and first point of fh)
        if self.fh.is_relative:
            if self.fh._values.values[0] != 1:
                raise ValueError(
                    "Relative fh must start with 1 and should not have any gaps."
                )
        # check that for ablosulte index/fh that there is no gap between cutoff and fh
        else:
            if not self._y.index[-1] == self.fh._values.shift(-1)[0]:
                raise ValueError(
                    "fh must start with next time point ater cutoff, no gap allowed."
                )
        shift = len(self.fh._values)
        if X.index.equals(self._y.index):
            y_t = self._y.copy()
            # shift values by fh into the future
            y_t.iloc[shift:] = y.iloc[:-shift]
            # replace last values with NaN
            y_t.iloc[:shift] = np.nan
        else:
            if len(X) != shift:
                raise ValueError("Given len of X must be equal to len of fh.")
            y_t = self._y.iloc[-shift:]
        col = self._y.name + "_" + self.suffix if self._y.name else self.suffix
        if col in X.columns:
            raise AttributeError(
                f"Name {col} is already in X.columns, please give a suffix param."
            )
        Xt[col] = self.transformer_.transform(y_t).values
        return Xt

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator."""
        return {
            "transformer": TabularToSeriesAdaptor(MinMaxScaler()),
            "fh": ForecastingHorizon([1, 2], is_relative=True),
        }
