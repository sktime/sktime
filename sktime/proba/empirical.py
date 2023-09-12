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
    ...     [[0, 1], [2, 3], [10, 11], [6, 7], [8, 9], [4, 5]],
    ...     index=spl_idx,
    ...     columns=["a", "b"],
    ... )
    >>> dist = Empirical(spl)
    >>> empirical_sample = dist.sample(3)
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
        _spl_instances = spl.index.droplevel(-1).unique()
        self._timestamps = _timestamps
        self._spl_instances = _spl_instances
        self._N = len(_spl_instances)

        if index is None:
            index = pd.Index(_timestamps)

        if columns is None:
            columns = spl.columns

        super().__init__(index=index, columns=columns)

        # initialized sorted samples
        self._init_sorted()

    def _init_sorted(self):
        """Initialize sorted version of spl."""
        times = self._timestamps
        cols = self.columns

        sorted = {}
        weights = {}
        weights
        for t in times:
            sorted[t] = {}
            weights[t] = {}
            for col in cols:
                spl_t = self.spl.loc[(slice(None), t), col].values
                sorter = np.argsort(spl_t)
                spl_t_sorted = spl_t[sorter]
                sorted[t][col] = spl_t_sorted
                if self.weights is not None:
                    weights_t = self.weights.loc[(slice(None), t)].values
                    weights_t_sorted = weights_t[sorter]
                    weights[t][col] = weights_t_sorted
                else:
                    ones = np.ones(len(spl_t_sorted))
                    weights[t][col] = ones

        self._sorted = sorted
        self._weights = weights

    def _apply_per_ix(self, func, params, x=None):
        """Apply function per index."""
        sorted = self._sorted
        weights = self._weights

        if x is not None and hasattr(x, "index"):
            index = x.index
        else:
            index = self.index
        if x is not None and hasattr(x, "columns"):
            cols = x.columns
        else:
            cols = self.columns

        res = pd.DataFrame(index=index, columns=cols)
        for ix in index:
            for col in cols:
                spl_t = sorted[ix][col]
                weights_t = weights[ix][col]
                if x is None:
                    x_t = None
                elif hasattr(x, "loc"):
                    x_t = x.loc[ix, col]
                else:
                    x_t = x
                res.loc[ix, col] = func(spl=spl_t, weights=weights_t, x=x_t, **params)
        return res.convert_dtypes()

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
        energy = self._apply_per_ix(_energy_np, {"assume_sorted": True}, x=x)
        res = pd.DataFrame(energy.sum(axis=1), columns=["energy"])
        return res

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
            mean_df = spl.groupby(level=-1).mean()
        else:
            mean_df = spl.groupby(level=-1).apply(
                lambda x: np.average(x, weights=self.weights.loc[x.index], axis=0)
            )
            mean_df = pd.DataFrame(mean_df.tolist(), index=mean_df.index)
            mean_df.columns = spl.columns

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
        N = self._N
        if self.weights is None:
            var_df = spl.groupby(level=-1).var(ddof=0)
        else:
            mean = self.mean()
            means = pd.concat([mean] * N, axis=0, keys=self._spl_instances)
            var_df = spl.groupby(level=-1).apply(
                lambda x: np.average(
                    (x - means.loc[x.index]) ** 2,
                    weights=self.weights.loc[x.index],
                    axis=0,
                )
            )
            var_df = pd.DataFrame(
                var_df.tolist(), index=var_df.index, columns=spl.columns
            )
        return var_df

    def cdf(self, x):
        """Cumulative distribution function."""
        cdf_val = self._apply_per_ix(_cdf_np, {"assume_sorted": True}, x=x)
        return cdf_val

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        ppf_val = self._apply_per_ix(_ppf_np, {"assume_sorted": True}, x=p)
        return ppf_val

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

        for _ in range(n_samples):
            smpls_i = []
            for t in timestamps:
                spl_from = spl.loc[(slice(None), t), :]
                if weights is not None:
                    spl_weights = weights.loc[(slice(None), t)].values
                else:
                    spl_weights = None
                spl_time = spl_from.sample(n=1, replace=True, weights=spl_weights)
                spl_time = spl_time.droplevel(0)
                smpls_i.append(spl_time)
            spl_i = pd.concat(smpls_i, axis=0)
            smpls.append(spl_i)

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
            [[0, 1], [2, 3], [10, 11], [6, 7], [8, 9], [4, 5]],
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


def _energy_np(spl, x=None, weights=None, assume_sorted=False):
    r"""Compute sample energy, fast numpy based subroutine.

    Let :math:`X` be the random variable with support being
    values of `spl`, with probability weights `weights`.

    This function then returns :math:`\mathbb{E}[|X-Y|]`, with :math:`Y` an
    independent copy of :math:`X`, if `x` is `None`.

    If `x` is passed, returns :math:`\mathbb{E}[|X-x|]`.

    Parameters
    ----------
    spl : 1D np.ndarray
        empirical sample
    x : None or float, optional, default=None
        if None, computes self-energy, if float, computes energy wrt x
    weights : None or 1D np.ndarray, optional, default=None
        if None, computes unweighted energy, if 1D np.ndarray, computes weighted energy
        if not None, must be of same length as ``spl``, needs not be normalized
    assume_sorted : bool, optional, default=False
        if True, assumes that ``spl`` is sorted in ascending order

    Returns
    -------
    float
        energy as described above
    """
    if weights is None:
        weights = np.ones(len(spl))

    if not assume_sorted:
        sorter = np.argsort(spl)
        spl = spl[sorter]
        weights = weights[sorter]

    w_sum = np.sum(weights)
    weights = weights / w_sum

    spl_diff = np.diff(spl)

    if x is None:
        cum_fwd = np.cumsum(weights[:-1])
        cum_back = np.cumsum(weights[1::-1])[::-1]
        energy = 2 * np.sum(cum_fwd * cum_back * spl_diff)
    else:
        spl_diff = np.abs(spl - x)
        energy = np.sum(weights * spl_diff)

    return energy


def _cdf_np(spl, x, weights=None, assume_sorted=False):
    """Compute empirical cdf, fast numpy based subroutine.

    Parameters
    ----------
    spl : 1D np.ndarray
        empirical sample
    x : float
        value at which to evaluate cdf
    weights : None or 1D np.ndarray, optional, default=None
        if None, computes unweighted cdf, if 1D np.ndarray, computes weighted cdf
        if not None, must be of same length as ``spl``, needs not be normalized
    assume_sorted : bool, optional, default=False
        if True, assumes that ``spl`` is sorted in ascending order

    Returns
    -------
    cdf_val float
        cdf-value at x
    """
    if weights is None:
        weights = np.ones(len(spl))

    if not assume_sorted:
        sorter = np.argsort(spl)
        spl = spl[sorter]
        weights = weights[sorter]

    w_sum = np.sum(weights)
    weights = weights / w_sum

    weights_select = weights[spl <= x]
    cdf_val = np.sum(weights_select)

    return cdf_val


def _ppf_np(spl, x, weights=None, assume_sorted=False):
    """Compute empirical ppf, fast numpy based subroutine.

    Parameters
    ----------
    spl : 1D np.ndarray
        empirical sample
    x : float
        probability at which to evaluate ppf
    weights : None or 1D np.ndarray, optional, default=None
        if None, computes unweighted ppf, if 1D np.ndarray, computes weighted ppf
        if not None, must be of same length as ``spl``, needs not be normalized
    assume_sorted : bool, optional, default=False
        if True, assumes that ``spl`` is sorted in ascending order

    Returns
    -------
    ppf_val float
        ppf-value at p
    """
    if weights is None:
        weights = np.ones(len(spl))

    if not assume_sorted:
        sorter = np.argsort(spl)
        spl = spl[sorter]
        weights = weights[sorter]

    w_sum = np.sum(weights)
    weights = weights / w_sum

    cum_weights = np.cumsum(weights)
    ix_val = np.searchsorted(cum_weights, x)
    ppf_val = spl[ix_val]

    return ppf_val
