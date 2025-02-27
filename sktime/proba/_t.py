# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Student's t-distribution."""

__author__ = ["Alex-JG3", "ivarzap"]

import numpy as np
import pandas as pd
from scipy.special import betaincinv, gamma, hyp2f1, loggamma

from sktime.proba.base import BaseDistribution


class TDistribution(BaseDistribution):
    """Student's t-distribution (sktime native).

    Parameters
    ----------
    mean : float or array of float (1D or 2D)
        mean of the t-distribution distribution
    sd : float or array of float (1D or 2D), must be positive
        standard deviation of the t-distribution distribution
    df : float or array of float (1D or 2D), must be positive
        Degrees of freedom of the t-distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> from sktime.proba.t import TDistribution

    >>> n = TDistribution(mu=[[0, 1], [2, 3], [4, 5]], sigma=1, df=10)
    """

    _tags = {
        "authors": ["Alex-JG3", "ivarzap"],
        "maintainers": ["Alex-JG3"],
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, mu, sigma, df=1, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.df = df
        self.index = index
        self.columns = columns

        self._mu, self._sigma, self._df = self._get_bc_params(
            self.mu, self.sigma, self.df
        )
        shape = self._mu.shape

        if index is None:
            index = pd.RangeIndex(shape[0])

        if columns is None:
            columns = pd.RangeIndex(shape[1])

        super().__init__(index=index, columns=columns)

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`. The expectation,
        :math:`\mathbb{E}[X]`, as infinite if :math:`\nu \le 1`.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        mean_arr = self._mu.copy()
        if (self._df <= 1).any():
            mean_arr = mean_arr.astype(np.float32)
            mean_arr[self._df <= 1] = np.inf
        return pd.DataFrame(mean_arr, index=self.index, columns=self.columns)

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns,

        .. math::
            \mathbb{V}[X] = \begin{cases}
                \frac{\nu}{\nu - 2} & \text{if} \nu > 2, \\
                \infty              & \text{if} \nu \le 2, \\
            \begin{cases}

        Where :math:`\nu` is the degrees of freedom of the t-distribution.

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        df_arr = self._df.copy()
        df_arr = df_arr.astype(np.float32)
        df_arr[df_arr <= 2] = np.inf
        mask = (df_arr > 2) & (df_arr != np.inf)
        df_arr[mask] = self._sigma[mask] ** 2 * df_arr[mask] / (df_arr[mask] - 2)
        return pd.DataFrame(df_arr, index=self.index, columns=self.columns)

    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        pdf_arr = gamma((d._df + 1) / 2)
        pdf_arr = pdf_arr / (np.sqrt(np.pi * d._df) * gamma(d._df / 2))
        pdf_arr = pdf_arr * (1 + ((x - d._mu) / d._sigma) ** 2 / d._df) ** (
            -(d._df + 1) / 2
        )
        pdf_arr = pdf_arr / d._sigma
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)

    def log_pdf(self, x):
        """Logarithmic probability density function."""
        d = self.loc[x.index, x.columns]
        lpdf_arr = loggamma((d._df + 1) / 2)
        lpdf_arr = lpdf_arr - 0.5 * np.log(d._df * np.pi)
        lpdf_arr = lpdf_arr - loggamma(d._df / 2)
        lpdf_arr = lpdf_arr - ((d._df + 1) / 2) * np.log(
            1 + ((x - d._mu) / d._sigma) ** 2 / d._df
        )
        lpdf_arr = lpdf_arr - np.log(d._sigma)
        return pd.DataFrame(lpdf_arr, index=x.index, columns=x.columns)

    def cdf(self, x):
        """Cumulative distribution function."""
        d = self.loc[x.index, x.columns]
        x_ = (x - d._mu) / d._sigma
        cdf_arr = x_ * gamma((d._df + 1) / 2)
        cdf_arr = cdf_arr * hyp2f1(0.5, (d._df + 1) / 2, 3 / 2, -(x_**2) / d._df)
        cdf_arr = 0.5 + cdf_arr / (np.sqrt(np.pi * d._df) * gamma(d._df / 2))
        return pd.DataFrame(cdf_arr, index=x.index, columns=x.columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        d = self.loc[p.index, p.columns]
        ppf_arr = p.to_numpy(copy=True)
        ppf_arr[p.values == 0.5] = 0.0
        ppf_arr[p.values <= 0] = -np.inf
        ppf_arr[p.values >= 1] = np.inf

        mask1 = (p.values < 0.5) & (p.values > 0)
        mask2 = (p.values < 1) & (p.values > 0.5)
        ppf_arr[mask1] = 1 / betaincinv(0.5 * d._df[mask1], 0.5, 2 * ppf_arr[mask1])
        ppf_arr[mask2] = 1 / betaincinv(
            0.5 * d._df[mask2], 0.5, 2 * (1 - ppf_arr[mask2])
        )
        ppf_arr[mask1 | mask2] = np.sqrt(ppf_arr[mask1 | mask2] - 1)
        ppf_arr[mask1 | mask2] = np.sqrt(d._df[mask1 | mask2]) * ppf_arr[mask1 | mask2]
        ppf_arr[mask1] = -ppf_arr[mask1]
        ppf_arr = d._sigma * ppf_arr + d._mu
        return pd.DataFrame(ppf_arr, index=p.index, columns=p.columns)

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
