"""Subsequence extraction transformer.

A transformer for the extraction of subsequences of specified length based on
maximal/minimal rolling window aggregates.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd

from sktime.transformations.base import BaseTransformer

__all__ = ["SubsequenceExtractionTransformer"]
__author__ = ["wirrywoo"]


class SubsequenceExtractionTransformer(BaseTransformer):
    """
    Extract subsequences of specified length based on rolling aggregatess.

    A transformer for the extraction of subsequences of specified length based on
    maximal/minimal rolling window aggregates.

    Parameters
    ----------
    subsequence_len : int
        Length of the subsequence. Must be less than the lengths of all input series.
    aggregate : {'mean', 'median'}, default 'mean'
        Function used to aggregate all values in subsequence to a scalar or primitive.
    method : {'max', 'min'}, default 'max'
        Function used to decide which subsequence to return from the set of scalars or
        primitives.

    Examples
    --------
    >>> from sktime.transformations.series.subsequence_extraction import (
    >>>     SubsequenceExtractionTransformer
    >>> )
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> X = _make_hierarchical(same_cutoff=False)
    >>> subseq_extract = SubsequenceExtractionTransformer(subsequence_len = 3)
    >>> subseq_extract.fit(X)
    >>> X_transformed = subseq_extract.transform(X)
    """

    _tags = {
        "univariate-only": False,
        "authors": ["wirrywoo"],
        "maintainers": ["wirrywoo"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": False,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
        "capability:inverse_transform": False,
        "capability:unequal_length:removes": True,
        "handles-missing-data": False,
    }

    def __init__(self, subsequence_len, aggregate="mean", method="max"):
        self.subsequence_len = subsequence_len
        self.aggregate = aggregate
        self.method = method

        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : reference to self
        """
        if self.subsequence_len > len(X):
            raise ValueError(
                f"Subsequence length parameter ({self.subsequence_len}) is not less \
                than or equal to the minimum sequence length of X ({len(X)})."
            )

        if self.aggregate not in ["mean", "median"]:
            raise ValueError(
                f"{self.aggregate} is currently not supported for parameter aggregate"
            )

        if self.method not in ["max", "min"]:
            raise ValueError(
                f"{self.method} is currently not supported for parameter method"
            )

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to transform
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series
            transformed version of X
        """
        index_list = X.index.get_level_values(X.index.names[-1])

        X_aggregate = getattr(
            X.rolling(window=self.subsequence_len), self.aggregate
        )().dropna()

        indices = getattr(X_aggregate, f"idx{self.method}")()

        upper = pd.Categorical(indices, categories=index_list, ordered=True).codes + 1
        lower = upper - self.subsequence_len

        dfs = [
            X[col].iloc[l:u].reset_index(drop=True)
            for col, l, u in zip(X.columns, lower, upper)
        ]

        return pd.concat(dfs, axis=1, ignore_index=False)

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
        params = [
            {"subsequence_len": 3},
            {"subsequence_len": 5, "aggregate": "median", "method": "min"},
        ]
        return params
