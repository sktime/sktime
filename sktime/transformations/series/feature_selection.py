#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements feature selection algorithms."""

__author__ = ["aiwalter"]
__all__ = ["FeatureSelection"]

import math

import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.forecasting import check_regressor


class FeatureSelection(BaseTransformer):
    """Select exogenous features.

    Transformer to enable tuneable feature selection of exogenous data. The
    FeatureSelection implements multiple methods to select features (columns).
    In case X is a pd.Series, then it is just passed through, unless method="none",
    then None is returned in transform().

    Parameters
    ----------
    method : str, required
        The method of how to select the features. Implemented methods are:

        * "feature-importances": Use feature_importances_ of the regressor (meta-model)
          to select n_columns with highest importance values.
          Requires parameter n_columns.
        * "random": Randomly select n_columns features. Requires parameter n_columns.
        * "columns": Select features by given names.
        * "none": Remove all columns, transform returns None.
        * "all": Select all given features.

    regressor : sklearn-like regressor, optional, default=None.
        Used as meta-model for the method "feature-importances". The given
        regressor must have an attribute "feature_importances_". If None,
        then a GradientBoostingRegressor(max_depth=5) is used.
    n_columns : int, optional
        Number of features (columns) to select. n_columns must be <=
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
        "authors": ["aiwalter"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,
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

        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to fit transform to
        y : pd.DataFrame, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self: a fitted instance of the estimator
        """
        self.n_columns_ = self.n_columns
        self.feature_importances_ = None

        # multivariate X
        if not isinstance(X, pd.Series):
            if self.method == "feature-importances":
                self.regressor_ = check_regressor(
                    regressor=self.regressor, random_state=self.random_state
                )
                self._check_n_columns(X)
                # fit regressor with X as exog data and y as endog data (target)
                self.regressor_.fit(X=X, y=y)
                if not hasattr(self.regressor_, "feature_importances_"):
                    raise ValueError(
                        """The given regressor must have an
                        attribute feature_importances_ after fitting."""
                    )
                # create dict with columns name (key) and feauter importance (value)
                d = dict(zip(X.columns, self.regressor_.feature_importances_))
                # sort d descending
                d = {k: d[k] for k in sorted(d, key=d.get, reverse=True)}
                self.feature_importances_ = d
                self.columns_ = list(d.keys())[: self.n_columns_]
            elif self.method == "random":
                self._check_n_columns(X)
                self.columns_ = list(
                    X.sample(
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
                self.columns_ = list(X.columns)
            else:
                raise ValueError("Incorrect method given. Try another method.")
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.Series or pd.DataFrame
            Data to be transformed
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : pd.Series or pd.DataFrame, same type as X
            transformed version of X
        """
        # multivariate case
        if not isinstance(X, pd.Series):
            if self.method == "none":
                Xt = None
            else:
                Xt = X[self.columns_]
        # univariate case
        else:
            if self.method == "none":
                Xt = None
            else:
                Xt = X
        return Xt

    def _check_n_columns(self, Z):
        if not isinstance(self.n_columns_, int):
            self.n_columns_ = int(math.ceil(Z.shape[1] / 2))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return {"method": "all"}
