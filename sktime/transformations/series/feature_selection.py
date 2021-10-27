#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements feature selection algorithms."""

__author__ = ["aiwalter"]
__all__ = ["FeatureSelection"]

import math
import pandas as pd

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series
from sktime.utils.validation.forecasting import check_regressor


class FeatureSelection(_SeriesToSeriesTransformer):
    """Select exogenous features.

    Transformer to enable tuneable feauture selection of exogenous data. The
    FeatureSelection implements multiple methods to select features (columns).
    In case Z is a pd.Series, then it is just passed through, unless method="none",
    then None is returned in transform().

    Parameters
    ----------
    method : str, required
        The method of how to select the features. Implemeted methods are:
        * "feature-importances": Use feature_importances_ of the regressor (meta-model)
          to select n_columns with highest importance values.
          Requires parameter n_columns.
        * "random": Randomly select n_columns features. Requires parameter n_columns.
        * "columns": Select features by given names.
        * "none": Remove all columns by setting Z to None.
        * "all": Select all given features.
    regressor : sklearn-like regressor, optional, default=None.
        Used as meta-model for the method "feature-importances". The given
        regressor must have an attribute "feature_importances_". If None,
        then a GradientBoostingRegressor(max_depth=5) is used.
    n_columns : int, optional
        Number of feautres (columns) to select. n_columns must be <=
        number of X columns. Some methods require n_columns to be given.
    random_state : int, RandomState instance or None, default=None
        Used to set random_state of the default regressor and to
        set random.seed() if method="random".
    columns : list of str
        A list of columns to select. If columns is given.

    Attributes
    ----------
    columns_ : list of str
        List of columns that have been selected as features.
    regressor_ : sklearn-like regressor
        Fitted regressor (meta-model).
    n_columns_: int
        Derived from number of features if n_columns is None, then
        n_columns_ is calculated as int(math.ceil(Z.shape[1] / 2)). So taking
        half of given features only as default.
    feature_importances_ : dict or None
        A dictionary with column name as key and feature imporatnce value as value.
        The dict is sorted descending on value. This attribute is a dict if
        method="feature-importances", else None.

    Examples
    --------
    >>> from sktime.transformations.series.feature_selection import FeatureSelection
    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()
    >>> transformer = FeatureSelection(method="feature-importances", n_columns=3)
    >>> Xt = transformer.fit_transform(X, y)
    """

    _tags = {
        "fit-in-transform": False,
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
        "univariate-only": False,
    }

    def __init__(
        self,
        method="feature-importances",
        n_columns=None,
        regressor=None,
        random_state=None,
        columns=None,
    ):
        self.n_columns = n_columns
        self.method = method
        self.regressor = regressor
        self.random_state = random_state
        self.columns = columns

        super(FeatureSelection, self).__init__()

    def fit(self, Z, X=None):
        """Fit the transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.Series, pd.DataFrame
            A time series to apply the transformation on.
        X : pd.DataFrame, default=None
            Exogenous variables are usd in method="feature-importances"
            to fit the meta-model (regressor).

        Returns
        -------
        self
        """
        Z = check_series(Z)
        self.n_columns_ = self.n_columns
        self.feature_importances_ = None

        # multivariate Z
        if not isinstance(Z, pd.Series):
            if self.method == "feature-importances":
                self.regressor_ = check_regressor(
                    regressor=self.regressor, random_state=self.random_state
                )
                self._check_n_columns(Z)
                X = check_series(X)
                # fit regressor with Z as exog data and X as endog data (target)
                self.regressor_.fit(X=Z, y=X)
                if not hasattr(self.regressor_, "feature_importances_"):
                    raise ValueError(
                        """The given regressor must have an
                        attribute feature_importances_ after fitting."""
                    )
                # create dict with columns name (key) and feauter importance (value)
                d = dict(zip(Z.columns, self.regressor_.feature_importances_))
                # sort d descending
                d = {k: d[k] for k in sorted(d, key=d.get, reverse=True)}
                self.feature_importances_ = d
                self.columns_ = list(d.keys())[: self.n_columns_]
            elif self.method == "random":
                self._check_n_columns(Z)
                self.columns_ = list(
                    Z.sample(
                        n=self.n_columns_, random_state=self.random_state, axis=1
                    ).columns
                )
            elif self.method == "columns":
                if self.columns is None:
                    raise AttributeError("Parameter columns must be given.")
                self.columns_ = self.columns
            elif self.method == "none":
                self.columns_ = None
            elif self.method == "all":
                self.columns_ = list(Z.columns)
            else:
                raise ValueError("Incorrect method given. Try another method.")

        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Return transformed version of input series `Z`.

        Parameters
        ----------
        Z : pd.Series, pd.DataFrame
            A time series to apply the transformation on.
        X : pd.DataFrame, default=None
            Exogenous data is ignored in transform.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            Transformed version of input series `Z`.
        """
        self.check_is_fitted()
        Z = check_series(Z)

        # multivariate case
        if not isinstance(Z, pd.Series):
            if self.method == "none":
                Zt = None
            else:
                Zt = Z[self.columns_]
        # univariate case
        else:
            if self.method == "none":
                Zt = None
            else:
                Zt = Z
        return Zt

    def _check_n_columns(self, Z):
        if not isinstance(self.n_columns_, int):
            self.n_columns_ = int(math.ceil(Z.shape[1] / 2))
