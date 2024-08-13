"""Subsequence extraction transformer - extract subsequences of specified length that
meet some criterion with respect to an aggregate function."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.transformations.panel.padder import PaddingTransformer

__all__ = ["SubsequenceExtractionTransformer"]
__author__ = ['wirrywoo']


class SubsequenceExtractionTransformer(BaseTransformer):
    """
    Parameters
    ----------
    subsequence_len : int
        Length of the subsequence. Must be less than the lengths of all input series.
    aggregate : {'mean', 'median'}, default 'mean'
        Function used to aggregate all values in subsequence to a scalar or primitive.
    method : {'max', 'min'}, default 'max'
        Function used to decide which subsequence to return from the set of scalars or primitives.

    Examples
    --------
    >>> from sktime.transformations.panel.subsequence_extraction import SubsequenceExtractionTransformer
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

        self.X_padded = PaddingTransformer(fill_value=np.nan).fit_transform(X)
        self.X_aggregate = self.X_padded.rolling(window = self.subsequence_len)

        if self.aggregate == "mean":
            self.X_aggregate = self.X_aggregate.mean()
        elif self.aggregate == "median":
            self.X_aggregate = self.X_aggregate.median()
        else:
            raise ValueError(f"{self.aggregate} is currently not supported for parameter aggregate")

        try:
            if self.method == "max":
                self.indices = self.X_aggregate.dropna().idxmax()
            elif self.method == "min":
                self.indices = self.X_aggregate.dropna().idxmin()
            else:
                raise ValueError(f"{self.method} is currently not supported for parameter method")
        except ValueError:
            raise ValueError(f"Subsequence length parameter ({self.subsequence_len}) is not less than minimum sequence length of X ({len(X)}).")

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
        
        upper = self.indices + 1
        lower = upper - self.subsequence_len

        dfs = [X[col].iloc[l:u].reset_index(drop=True) 
            for col, l, u in zip(X.columns, lower, upper)]
        
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

        params = {"subsequence_len": 3}
        return params
