# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test class for default methods.

This is not for direct use, but for testing whether the defaulting in various methods
works.

Testing works via TestAllDistributions which discovers the classes in here, executes the
public methods in interface conformance tests, which in turn triggers the fallback
defaults.
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from scipy.special import erfinv

from sktime.proba.base import BaseDistribution


# normal distribution with exact implementations removed
class _DistrDefaultMethodTester(BaseDistribution):
    """Tester distribution for default methods."""

    _tags = {
        "capabilities:approx": ["pdfnorm", "mean", "var", "energy", "log_pdf", "cdf"],
        "capabilities:exact": ["pdf", "ppf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, mu, sigma, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.index = index
        self.columns = columns

        self._mu, self._sigma = self._get_bc_params(self.mu, self.sigma)
        shape = self._mu.shape

        if index is None:
            index = pd.RangeIndex(shape[0])

        if columns is None:
            columns = pd.RangeIndex(shape[1])

        super().__init__(index=index, columns=columns)

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        d = self.loc[p.index, p.columns]
        icdf_arr = d.mu + d.sigma * np.sqrt(2) * erfinv(2 * p.values - 1)
        return pd.DataFrame(icdf_arr, index=p.index, columns=p.columns)

    def pdf(self, x):
        """Probability density function."""
        d = self.loc[x.index, x.columns]
        pdf_arr = np.exp(-0.5 * ((x.values - d.mu) / d.sigma) ** 2)
        pdf_arr = pdf_arr / (d.sigma * np.sqrt(2 * np.pi))
        return pd.DataFrame(pdf_arr, index=x.index, columns=x.columns)
