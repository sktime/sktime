# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Probability distribution objects with tensorflow-probability back-end."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from sktime.proba.base import _BaseTFDistribution
from sktime.utils.validation._dependencies import _check_estimator_deps


class TFNormal(_BaseTFDistribution):
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
    >>> from sktime.proba.tfp import TFNormal  # doctest: +SKIP

    >>> n = Normal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)  # doctest: +SKIP
    """

    _tags = {
        "python_dependencies": "tensorflow_probability",
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
    }

    def __init__(self, mu, sigma, index=None, columns=None):

        self.mu = mu
        self.sigma = sigma

        _check_estimator_deps(self)

        import tensorflow_probability as tfp

        tfd = tfp.distributions

        mu, sigma = self._get_bc_params()
        distr = tfd.Normal(loc=mu, scale=sigma)

        if index is None:
            index = pd.RangeIndex(distr.batch_shape[0])

        if columns is None:
            columns = pd.RangeIndex(distr.batch_shape[1])

        super(TFNormal, self).__init__(index=index, columns=columns, distr=distr)

    def _get_bc_params(self):
        """Fully broadcast parameters of self, given param shapes and index, columns."""
        to_broadcast = [self.mu, self.sigma]
        if hasattr(self, "index"):
            to_broadcast += [self.index.to_numpy().reshape(-1, 1)]
        if hasattr(self, "columns"):
            to_broadcast += [self.index.to_numpy()]
        bc = np.broadcast_arrays(*to_broadcast)
        return bc[0], bc[1]

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
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        mean_arr, _ = np.broadcast_arrays(self.mu, self.sigma)
        return pd.DataFrame(mean_arr, index=self.index, columns=self.columns)

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
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
