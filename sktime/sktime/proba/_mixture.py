# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mixture distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.base._meta import _HeterogenousMetaEstimator
from sktime.proba.base import BaseDistribution


class Mixture(_HeterogenousMetaEstimator, BaseDistribution):
    """Mixture of distributions.

    Parameters
    ----------
    distributions : list of tuples (str, BaseDistribution) or BaseDistribution
        list of mixture components
    weights : list of float, optional, default = None
        list of mixture weights, will be normalized to sum to 1
        if not provided, uniform mixture is assumed
    index : pd.Index, optional, default = inferred from component distributions
    columns : pd.Index, optional, default = inferred from component distributions

    Example
    -------
    >>> from sktime.proba.mixture import Mixture
    >>> from sktime.proba.normal import Normal
    >>>
    >>> n1 = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)
    >>> n2 = Normal(mu=3, sigma=2, index=n1.index, columns=n1.columns)
    >>> m = Mixture(distributions=[("n1", n1), ("n2", n2)], weights=[0.3, 0.7])
    >>> mixture_sample = m.sample(n_samples=10)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm", "energy", "ppf"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf"],
        "distr:measuretype": "mixed",
    }

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_distributions"

    def __init__(self, distributions, weights=None, index=None, columns=None):
        self.distributions = distributions
        self.weights = weights
        self.index = index
        self.columns = columns

        self._distributions = distributions
        n_dists = len(self._distributions)

        if weights is None:
            self._weights = np.ones(n_dists) / n_dists
        else:
            self._weights = np.array(weights) / np.sum(weights)

        if index is None:
            index = self._distributions[0][1].index

        if columns is None:
            columns = self._distributions[0][1].columns

        super().__init__(index=index, columns=columns)

    def _iloc(self, rowidx=None, colidx=None):
        dists = self._distributions
        weights = self.weights

        dists_subset = [(x[0], x[1].iloc[rowidx, colidx]) for x in dists]

        index_subset = dists_subset[0][1].index
        columns_subset = dists_subset[0][1].columns

        return Mixture(
            distributions=dists_subset,
            weights=weights,
            index=index_subset,
            columns=columns_subset,
        )

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        return self._average("mean")

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        weights = self._weights
        var_mean = self._average("var")
        mixture_mean = self._average("mean")

        means = [d.mean() for _, d in self._distributions]
        mean_var = [(m - mixture_mean) ** 2 for m in means]
        var_mean_var = self._average_df(mean_var, weights=weights)

        return var_mean + var_mean_var

    def _average(self, method, x=None, weights=None):
        """Average a method over the mixture components."""
        if x is None:
            args = ()
        else:
            args = (x,)

        vals = [getattr(d, method)(*args) for _, d in self._distributions]

        return self._average_df(vals, weights=weights)

    def _average_df(self, df_list, weights=None):
        """Average a list of `pd.DataFrame` objects, with weights."""
        if weights is None and hasattr(self, "_weights"):
            weights = self._weights
        elif weights is None:
            weights = np.ones(len(df_list)) / len(df_list)

        n_df = len(df_list)
        df_weighted = [df * w for df, w in zip(df_list, weights)]
        df_concat = pd.concat(df_weighted, axis=1, keys=range(n_df))
        df_res = df_concat.T.groupby(level=-1).sum().T
        return df_res

    def pdf(self, x):
        """Probability density function."""
        return self._average("pdf", x)

    def cdf(self, x):
        """Cumulative distribution function."""
        return self._average("cdf", x)

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
        if n_samples is None:
            N = 1
        else:
            N = n_samples

        n_dist = len(self._distributions)
        selector = np.random.choice(n_dist, size=N, p=self._weights)

        samples = [self._distributions[i][1].sample() for i in selector]

        if n_samples is None:
            return samples[0]
        else:
            return pd.concat(samples, axis=0, keys=range(N))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.proba.normal import Normal

        index = pd.RangeIndex(3)
        columns = pd.Index(["a", "b"])
        normal1 = Normal(mu=0, sigma=1, index=index, columns=columns)
        normal2 = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, columns=columns)

        dists = [("normal1", normal1), ("normal2", normal2)]

        params1 = {"distributions": dists}
        params2 = {"distributions": dists, "weights": [0.3, 0.7]}
        return [params1, params2]
