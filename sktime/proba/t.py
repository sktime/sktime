# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Student's t-distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from scipy.special import erfinv, gamma, hyp2f1

from sktime.proba.base import BaseDistribution


class TDistribution(BaseDistribution):
    """Student's t-distribution (sktime native).

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
    >>> from sktime.proba.t import TDistribution

    >>> n = TDistribution(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, df=10)
    """

    _tags = {
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, mu, sigma, df=1, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.df = df
        self.index = index
        self.columns = columns

        self._mu, self._sigma, self._df = self._get_bc_params()
        shape = self._mu.shape

        if index is None:
            index = pd.RangeIndex(shape[0])

        if columns is None:
            columns = pd.RangeIndex(shape[1])

        super().__init__(index=index, columns=columns)

    def _get_bc_params(self):
        """Fully broadcast parameters of self, given param shapes and index, columns."""
        to_broadcast = [self.mu, self.sigma, self.df]
        if hasattr(self, "index") and self.index is not None:
            to_broadcast += [self.index.to_numpy().reshape(-1, 1)]
        if hasattr(self, "columns") and self.columns is not None:
            to_broadcast += [self.columns.to_numpy()]
        bc = np.broadcast_arrays(*to_broadcast)
        return bc[0], bc[1], bc[2]

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
        Returns,

        .. math::
            \mathbb{V}[X] = \begin{cases}
                \frac{\nu}{\nu - 2} & \text{if} \nu > 2, \\
                \infty              & \text{if} \nu = 2, \\
                \text{NaN}          & \text{if} \nu < 2.
            \begin{cases}

        Where :math:`\nu` is the degrees of freedom of the t-distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        df_arr = self._df.copy()
        df_arr = df_arr.astype(np.float32)
        df_arr[df_arr < 2] = np.nan
        df_arr[df_arr == 2] = np.inf
        mask = (df_arr > 2) & (df_arr != np.inf)
        df_arr[mask] = df_arr[mask] / (df_arr[mask] - 2)
        return pd.DataFrame(df_arr, index=self.index, columns=self.columns)

    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        pdf_arr = gamma((d.df + 1) / 2)
        pdf_arr = pdf_arr / (np.sqrt(np.pi * d.df) * gamma(d.df / 2))
        pdf_arr = pdf_arr * (1 + x**2 / d.df) ** (-(d.df + 1) / 2)
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pdf(self, x):
        """Logarithmic probability density function."""
        d = self.loc[x.index, x.columns]
        lpdf_arr = -0.5 * ((x.values - d.mu) / d.sigma) ** 2
        lpdf_arr = lpdf_arr - np.log(d.sigma * np.sqrt(2 * np.pi))
        return pd.DataFrame(lpdf_arr, index=x.index, columns=x.columns)

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        cdf_arr = x * gamma((d.df + 1) / 2)
        cdf_arr = cdf_arr * hyp2f1(0.5, (d.df + 1) / 2, 3 / 2, -(x**2) / d.df)
        cdf_arr = 0.5 + cdf_arr / (np.sqrt(np.pi * d.df) * gamma(d.df / 2))
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        d = self.loc[p.index, p.columns]
        icdf_arr = d.mu + d.sigma * np.sqrt(2) * erfinv(2 * p.values - 1)
        return pd.DataFrame(icdf_arr, index=p.index, columns=p.columns)

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
