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

    >>> n = TFNormal(mu=[[0, 1], [2, 3], [4, 5]], sigma=1)  # doctest: +SKIP
    """

    _tags = {
        "python_dependencies": "tensorflow_probability",
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.index = index
        self.columns = columns

        _check_estimator_deps(self)

        import tensorflow_probability as tfp

        tfd = tfp.distributions

        # todo: untangle index handling
        # and broadcast of parameters.
        # move this functionality to the base class
        # 0.19.0?
        self._mu, self._sigma = self._get_bc_params(self.mu, self.sigma, dtype="float")
        distr = tfd.Normal(loc=self._mu, scale=self._sigma)
        shape = self._mu.shape

        if index is None:
            index = pd.RangeIndex(shape[0])

        if columns is None:
            columns = pd.RangeIndex(shape[1])

        super().__init__(index=index, columns=columns, distr=distr)

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
            sd_arr = self._sigma
            energy_arr = 2 * np.sum(sd_arr, axis=1) / np.sqrt(np.pi)
            energy = pd.DataFrame(energy_arr, index=self.index, columns=["energy"])
        else:
            mu_arr, sd_arr = self._mu, self._sigma
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
        mean_arr = self._mu
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
        sd_arr = self._sigma
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
