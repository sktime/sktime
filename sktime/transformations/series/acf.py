#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Auto-correlation transformations.

Module :mod:`sktime.transformations.series` implements auto-correlation
transformers.
"""

__author__ = ["Afzal Ansari"]
__all__ = ["AutoCorrelationTransformer", "PartialAutoCorrelationTransformer"]

import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class AutoCorrelationTransformer(_SeriesToSeriesTransformer):
    """Auto-correlation transformer.

    The autocorrelation function measures how correlated a timeseries is
    with itself at different lags. The AutocorrelationTransformer returns
    these values as a series for each lag up to the `n_lags` specified.

    Parameters
    ----------
    adjusted : bool, default=False
        If True, then denominators for autocovariance are n-k, otherwise n.

    n_lags : int, default=None
        Number of lags to return autocorrelation for. If None,
        statsmodels acf function uses min(10 * np.log10(nobs), nobs - 1).

    fft : bool, default=False
        If True, computes the ACF via FFT.

    missing : {"none", "raise", "conservative", "drop"}, default="none"
        How missing values are to be treated in autocorrelation function
        calculations.

        - "none" performs no checks or handling of missing values
        - “raise” raises an exception if NaN values are found.
        - “drop” removes the missing observations and then estimates the
          autocovariances treating the non-missing as contiguous.
        - “conservative” computes the autocovariance using nan-ops so that nans
          are removed when computing the mean and cross-products that are used to
          estimate the autocovariance. "n" in calculation is set to the number of
          non-missing observations.

    See Also
    --------
    PartialAutoCorrelationTransformer

    Notes
    -----
    Provides wrapper around statsmodels
    `acf <https://www.statsmodels.org/devel/generated/
    statsmodels.tsa.stattools.acf.html>`_ function.

    Examples
    --------
    >>> from sktime.transformations.series.acf import AutoCorrelationTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = AutoCorrelationTransformer(n_lags=12)
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {"univariate-only": True, "fit-in-transform": True}

    def __init__(
        self,
        adjusted=False,
        n_lags=None,
        fft=False,
        missing="none",
    ):
        self.adjusted = adjusted
        self.n_lags = n_lags
        self.fft = fft
        self.missing = missing
        super(AutoCorrelationTransformer, self).__init__()

    def transform(self, Z, X=None):
        """Transform data.

        Parameters
        ----------
        Z : pd.Series
            Series to transform
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation

        Returns
        -------
        Zt : pd.Series
            Transformed series
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)

        # Passing an alpha values other than None would return confidence intervals
        # and break the signature of the series-to-series transformer
        zt = acf(
            z,
            adjusted=self.adjusted,
            nlags=self.n_lags,
            qstat=False,
            fft=self.fft,
            alpha=None,
            missing=self.missing,
        )
        return pd.Series(zt)


class PartialAutoCorrelationTransformer(_SeriesToSeriesTransformer):
    """Partial auto-correlation transformer.

    The partial autocorrelation function measures the conditional correlation
    between a timeseries and its self at different lags. In particular,
    the correlation between a time period and a lag, is calculated conditional
    on all the points between the time period and the lag.

    The PartialAutoCorrelationTransformer returns
    these values as a series for each lag up to the `n_lags` specified.

    Parameters
    ----------
    n_lags : int, default=None
        Number of lags to return partial autocorrelation for. If None,
        statsmodels acf function uses min(10 * np.log10(nobs), nobs // 2 - 1).

    method : str, default="ywadjusted"
        Specifies which method for the calculations to use.

        - "yw" or "ywadjusted" : Yule-Walker with sample-size adjustment in
          denominator for acovf. Default.
        - "ywm" or "ywmle" : Yule-Walker without adjustment.
        - "ols" : regression of time series on lags of it and on constant.
        - "ols-inefficient" : regression of time series on lags using a single
          common sample to estimate all pacf coefficients.
        - "ols-adjusted" : regression of time series on lags with a bias
          adjustment.
        - "ld" or "ldadjusted" : Levinson-Durbin recursion with bias
          correction.
        - "ldb" or "ldbiased" : Levinson-Durbin recursion without bias
          correction.

    See Also
    --------
    AutoCorrelationTransformer

    Notes
    -----
    Provides wrapper around statsmodels
    `pacf <https://www.statsmodels.org/devel/generated/
    statsmodels.tsa.stattools.pacf.html>`_ function.


    Examples
    --------
    >>> from sktime.transformations.series.acf import PartialAutoCorrelationTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = PartialAutoCorrelationTransformer(n_lags=12)
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {"univariate-only": True, "fit-in-transform": True}

    def __init__(
        self,
        n_lags=None,
        method="ywadjusted",
    ):
        self.n_lags = n_lags
        self.method = method
        super(PartialAutoCorrelationTransformer, self).__init__()

    def transform(self, Z, X=None):
        """Transform data.

        Parameters
        ----------
        Z : pd.Series
            Series to transform
        X : pd.DataFrame, optional (default=None)
            Exogenous data used in transformation

        Returns
        -------
        Zt : pd.Series
            Transformed series
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)

        # Passing an alpha values other than None would return confidence intervals
        # and break the signature of the series-to-series transformer
        zt = pacf(z, nlags=self.n_lags, method=self.method, alpha=None)
        return pd.Series(zt)
