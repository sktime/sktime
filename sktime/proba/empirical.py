# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Empirical distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.proba.base import BaseDistribution


class Empirical(BaseDistribution):
    """Empirical distribution (sktime native).

    Parameters
    ----------
    spl : pd.DataFrame with pd.MultiIndex
        empirical sample
        last (highest) index is time, first (lowest) index is sample
    weights : pd.Series, with same index and length as spl, optional, default=None
        if not passed, ``spl`` is assumed to be unweighted
    time_indep : bool, optional, default=True
        if True, ``sample`` will sample individual time indices independently
        if False, ``sample`` will sample etire instances from ``spl``
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> import pandas as pd
    >>> from sktime.proba.empirical import Empirical

    >>> spl_idx = pd.MultiIndex.from_product(
    ...     [[0, 1], [0, 1, 2]], names=["sample", "time"]
    ... )
    >>> spl = pd.DataFrame(
    ...     [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    ...     index=spl_idx,
    ...     columns=["a", "b"],
    ... )
    >>> dist = Empirical(spl)
    >>> dist.sample(3)
    """

    _tags = {
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "energy", "cdf", "ppf"],
        "distr:measuretype": "discrete",
    }

    def __init__(self, spl, weights=None, time_indep=True, index=None, columns=None):

        self.spl = spl
        self.weights = weights
        self.time_indep = time_indep
        self.index = index
        self.columns = columns

        _timestamps = spl.index.get_level_values(-1).unique()
        self._timestamps = _timestamps

        shape0 = len(np.unique(_timestamps))
        shape = (shape0, spl.shape[1])

        if index is None:
            index = pd.RangeIndex(shape[0])

        if columns is None:
            columns = pd.RangeIndex(shape[1])

        super(Empirical, self).__init__(index=index, columns=columns)

    def energy(self, x=None):
        r"""Energy of self, w.r.t. self or a constant frame x.

        Let :math:`X, Y` be i.i.d. random variables with the distribution of `self`.

        If `x` is `None`, returns :math:`\mathbb{E}[|X-Y|]` (per row), "self-energy".
        If `x` is passed, returns :math:`\mathbb{E}[|X-x|]` (per row), "energy wrt x".

        Parameters
        ----------
        x : None or pd.DataFrame, optional, default=None
            if pd.DataFrame, must have same rows and columns as `self`

        Returns
        -------
        pd.DataFrame with same rows as `self`, single column `"energy"`
        each row contains one float, self-energy/energy as described above.
        """
        if x is None:
            return "todo"
        else:
            return "todo"

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        spl = self.spl
        if self.weights is None:
            mean_df = spl.groupby(level=0).mean()
        else:
            mean_df = spl.groupby(level=0).apply(
                lambda x: np.average(x, weights=self.weights.loc[x.index], axis=0)
            )
        return mean_df

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        spl = self.spl
        if self.weights is None:
            var_df = spl.groupby(level=0).var(ddof=0)
        else:
            var_df = spl.groupby(level=0).apply(
                lambda x: np.average(
                    (x - self.mean().loc[x.index])**2,
                    weights=self.weights.loc[x.index],
                    axis=0,
                )
            )
        return var_df

    def cdf(self, x):
        """Cumulative distribution function."""
        return "todo"

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        return "todo"

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None

        Returns
        -------
        if `n_samples` is `None`:
        returns a sample that contains a single sample from `self`,
        in `pd.DataFrame` mtype format convention, with `index` and `columns` as `self`
        if n_samples is `int`:
        returns a `pd.DataFrame` that contains `n_samples` i.i.d. samples from `self`,
        in `pd-multiindex` mtype format convention, with same `columns` as `self`,
        and `MultiIndex` that is product of `RangeIndex(n_samples)` and `self.index`
        """
        spl = self.spl
        timestamps = self._timestamps
        weights = self.weights

        if n_samples is None:
            n_samples = 1
            n_samples_was_none = True
        else:
            n_samples_was_none = False
        smpls = []

        for i in range(n_samples):
            smpls_i = []
            for t in timestamps:
                spl_from = spl.loc[(slice(None), t), :]
                if weights is not None:
                    spl_weights = weights.loc[(slice(None), t), :]
                else:
                    spl_weights = None
                spl_time = spl_from.sample(n=1, replace=True, weights=spl_weights)
                spl_time = spl_time.droplevel(0)
                smpls_i.append(spl_time)
            spl_i = pd.concat(smpls_i, axis=0)
            smpls.append(spl_i)
            print(spl_i)

        spl = pd.concat(smpls, axis=0, keys=range(n_samples))
        if n_samples_was_none:
            spl = spl.droplevel(0)

        return spl

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # params1 is a DataFrame with simple row multiindex
        spl_idx = pd.MultiIndex.from_product(
            [[0, 1], [0, 1, 2]], names=["sample", "time"]
        )
        spl = pd.DataFrame(
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
            index=spl_idx,
            columns=["a", "b"],
        )
        params1 = {
            "spl": spl,
            "weights": None,
            "time_indep": True,
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a", "b"]),
        }

        # params2 is weighted
        params2 = {
            "spl": spl,
            "weights": pd.Series([0.5, 0.5, 0.5, 1, 1, 1.1], index=spl_idx),
            "time_indep": False,
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
