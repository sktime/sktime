"""Subsequence extraction transformer.

A transformer for the extraction of contiguous subsequences of specified
length based on maximal/minimal rolling window aggregates.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import warnings
from functools import partial

import numpy as np
import pandas as pd
from numpy._core._multiarray_umath import _ArrayFunctionDispatcher

from sktime.transformations.base import BaseTransformer

__all__ = ["SubsequenceExtractionTransformer"]
__author__ = ["wirrywoo"]


class SubsequenceExtractionTransformer(BaseTransformer):
    r"""
    Extract contiguous subsequences of specified length based on rolling aggregates.

    A transformer for the extraction of contiguous subsequences of specified
    length based on maximal/minimal rolling window aggregates.

    Given a sequence :math:`\\{x_1, x_2, \cdots, x_n \\}` and ``subseq_len`` integer
    :math:`k` such that :math:`0 < k \leq n`, the transformer's task is to find index
    :math:`i` satisfying :math:`1 \leq i \leq i + k - 1 \leq n` such that for given
    ``aggregate_fn`` :math:`A: \mathbb{R}^k \longrightarrow \mathbb{R}`:

    1. :math:`A(x_{i}, \cdots, x_{i+k-1})` is maximal when ``selector = 'max'``, and
    2. :math:`A(x_{i}, \cdots, x_{i+k-1})` is minimal when ``selector = 'min'``.

    The `maximum sum subarray problem <https://en.wikipedia.org/wiki/Maximum_subarray_problem>`_
    is a special case and can be obtained by setting ``aggregate_fn = np.sum`` and
    ``selector = 'max'``.

    Parameters
    ----------
    aggregate_fn : callable of signature ``np.ndarray -> float``
        Callable function in ``numpy`` used to aggregate values in contiguous
        subsequence to a scalar.
    subseq_len : int
        Length of the subsequence in .iloc units. Must be less than the lengths of all
        input series.
    kwargs : dict, default: None
        Dictionary of additional keyword arguments to pass to aggregate_fn.
    selector : {'max', 'min'}, default: 'max'
        Function used to decide which subsequence to return from the set of scalars or
        primitives.

    Examples
    --------
    >>> from sktime.transformations.series.subsequence_extraction import (
    >>>     SubsequenceExtractionTransformer
    >>> )
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> X = _make_hierarchical(same_cutoff=False)
    >>> subseq_extract = SubsequenceExtractionTransformer(
    >>>     aggregate_fn = np.sum, subseq_len = 3)
    >>> subseq_extract.fit(X)
    >>> X_transformed = subseq_extract.transform(X)

    References
    ----------
    Jon Bentley. 1984. Programming pearls: algorithm design techniques.
    Commun. ACM 27, 9 (Sept. 1984), 865-873. https://doi.org/10.1145/358234.381162
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

    def __init__(self, aggregate_fn, subseq_len, kwargs=None, selector="max"):
        self.aggregate_fn = aggregate_fn
        self.subseq_len = subseq_len
        self.kwargs = kwargs
        self.selector = selector

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
        if self.subseq_len > len(X):
            raise ValueError(
                f"Subsequence length parameter ({self.subseq_len}) is not less \
                than or equal to the minimum sequence length of X ({len(X)})."
            )

        if not (isinstance(self.aggregate_fn, _ArrayFunctionDispatcher)):
            raise ValueError(
                f"{self.aggregate_fn} is not supported for parameter aggregate_fn"
            )

        if self.selector not in ["max", "min"]:
            raise ValueError(f"{self.selector} is not supported for parameter selector")

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

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            fnc = partial(self.aggregate_fn, **(self.kwargs or {}))
            X_aggregate = getattr(X.rolling(window=self.subseq_len), "agg")(
                fnc.func, **fnc.keywords
            ).dropna()

        indices = getattr(X_aggregate, f"idx{self.selector}")()

        upper = pd.Categorical(indices, categories=index_list, ordered=True).codes + 1
        lower = upper - self.subseq_len

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
            {
                "subseq_len": 3,
                "aggregate_fn": np.average,
                "kwargs": {"weights": [0.5, 0.3, 0.2], "axis": 0},
                "selector": "max",
            },
            {"subseq_len": 5, "aggregate_fn": np.median, "selector": "min"},
            {"subseq_len": 8, "aggregate_fn": np.mean},
        ]
        return params
