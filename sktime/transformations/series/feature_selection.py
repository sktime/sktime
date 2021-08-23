#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements feature selection algorithms."""

__author__ = ["aiwalter"]
__all__ = ["FeatureSelection"]

from sklearn.base import clone
from sklearn.base import is_regressor
from sklearn.ensemble import GradientBoostingRegressor

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class FeatureSelection(_SeriesToSeriesTransformer):
    """Select exogenous features.

    Transformer to enable tuneable feauture selection of exogenous data. The
    FeatureSelection implements multiple methods to select features (columns).

    Parameters
    ----------
    method : str
        The method of how to select the features. Implemeted methods are:
        * "feature-importances": Use feature_importances_ of the regressor (meta-model)
          to select n_features with highest importance values.
        * "random": Randomly select n_features features.
        * "columns": Select features by given names.
    regressor : sklearn-like regressor, optional, default=None.
        Used as meta-model for the method "feature-importances". The given
        regressor must have an attribute "feature_importances_". If None,
        then a GradientBoostingRegressor(max_depth=5) is used.
    n_features : int
        Number of feautres (columns) to select. n_features must be <=
        number of X columns.
    random_state : int/float/str, optional
        Value to set random.seed() if method="random", default None
    columns : list of str
        A list of columns to select. If columns is given, n_features
        must be set to None.

    Attributes
    ----------
    columns_ : list of str
        List of columns that have been selected as features.
    regressor_ : sklearn-like regressor
        Fitted regressor (meta-model).

    Examples
    --------
    >>> from sktime.transformations.series.feature_selection import FeatureSelection
    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()
    >>> transformer = FeatureSelection(method="feature-importances", n_feautures=3)
    >>> Xt = transformer.fit_transform(X, y)
    """

    _tags = {
        "fit-in-transform": False,
        "transform-returns-same-time-index": True,
        "univariate-only": False,
        "skip-inverse-transform": True,
    }

    def __init__(
        self,
        method="feature-importances",
        n_features=2,
        regressor=None,
        random_state=None,
        columns=None,
    ):
        self.n_features = n_features
        self.method = method
        self.regressor = regressor
        self.random_state = random_state
        self.columns = columns

        super(FeatureSelection, self).__init__()

    def fit(self, Z, X=None):
        """Fit the transformation on input series `Z`.

        Parameters
        ----------
        Z : pd.DataFrame
            A time series to apply the transformation on.

        Returns
        -------
        self
        """
        Z = check_series(Z, enforce_multivariate=True)

        if self.method == "feature-importances":
            self._check_regressor()
            _check_n_features(self.n_features)
            X = check_series(X)
            # fit regressor with Z as exog data and X as endog data (target)
            self.regressor_.fit(X=Z, y=X)
            # create dirctionary with columns name (key) and feauter importance (value)
            d = dict(zip(Z.columns, self.regressor_.feature_importances_))
            # sort d descending
            d = {k: d[k] for k in sorted(d, key=d.get, reverse=True)}
            self.columns_ = list(d.keys())[: self.n_features]
        elif self.method == "random":
            _check_n_features(self.n_features)
            self.columns_ = list(
                Z.sample(
                    n=self.n_features, random_state=self.random_state, axis=1
                ).columns
            )
        elif self.method == "columns":
            if self.columns is None:
                raise AttributeError("Parameter columns must be given.")
            self.columns_ = self.columns
        else:
            raise ValueError("Incorrect method given. Try another method.")

        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        """Return transformed version of input series `Z`.

        Parameters
        ----------
        Z : pd.DataFrame
            A time series to apply the transformation on.

        Returns
        -------
        Zt : pd.Series or pd.DataFrame
            Transformed version of input series `Z`.
        """
        self.check_is_fitted()
        Z = check_series(Z, enforce_multivariate=True)

        Zt = Z[self.columns_]

        return Zt

    def _check_regressor(self):
        if self.regressor is None:
            self.regressor_ = GradientBoostingRegressor(max_depth=5)
        else:
            if not is_regressor(self.regressor):
                raise ValueError(
                    f"`regressor` should be a regressor, "
                    f"but found: {self.regressor}"
                )
            self.regressor_ = clone(self.regressor)
        return self


def _check_n_features(n_features):
    if not isinstance(n_features, int):
        raise ValueError("Parameter n_features must be int")
