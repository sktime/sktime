# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.proba.base import _BaseTFDistribution
from sktime.utils.validation._dependencies import _check_estimator_deps


class Normal(_BaseTFDistribution):
    """Normal distribution with tensorflow-probability back-end.

    Parameters
    ----------
    mean : float or array of float (1D or 2D)
        mean of the normal distribution
    sd : float or array of float (1D or 2D), must be positive
        standard deviation of the normal distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from sktime.proba.tfp import Normal  # doctest: +SKIP

    >>> n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)  # doctest: +SKIP
    """

    _tags = {"python_dependencies": "tensorflow_probability"}

    def __init__(self, mu, sigma, index=None, columns=None):

        self.mu = mu
        self.sigma = sigma

        _check_estimator_deps(self)

        import tensorflow_probability as tfp

        tfd = tfp.distributions

        distr = tfd.Normal(loc=mu, scale=sigma)

        if index is None:
            index = pd.RangeIndex(distr.batch_shape[0])

        if columns is None:
            columns = pd.RangeIndex(distr.batch_shape[1])

        super(Normal, self).__init__(index=index, columns=columns, distr=distr)

    def energy(self, x=None):
        """Energy of self, w.r.t. self or a constant frame x."""
        # note: self-energy, x=None case seems correct
        if x is None:
            _, sd_arr = np.broadcast_arrays(self.mu, self.sigma)
            energy_arr = 2 * np.sum(sd_arr, axis=1) / np.sqrt(np.pi)
            energy = pd.DataFrame(energy_arr, index=self.index, columns=["energy"])
        else:
            mu_arr, sd_arr = np.broadcast_arrays(self.mu, self.sigma)
            c_arr = (x - mu_arr) * (2 * self.cdf(x) - 1) + 2 * sd_arr**2 * self.pdf(x)
            energy_arr = np.sum(c_arr, axis=1)
            energy = pd.DataFrame(energy_arr, index=self.index, columns=["energy"])
        return energy

    def mean(self):
        """Return expected value of the distribution."""
        mean_arr, _ = np.broadcast_arrays(self.mu, self.sigma)
        return pd.DataFrame(mean_arr, index=self.index, columns=self.columns)

    def var(self):
        """Return element/entry-wise variance of the distribution."""
        _, sd_arr = np.broadcast_arrays(self.mu, self.sigma)
        return pd.DataFrame(sd_arr, index=self.index, columns=self.columns) ** 2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[0, 1], [2, 3], [4, 5]], "sigma": 1}
        params2 = {
            "mu": 0,
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
