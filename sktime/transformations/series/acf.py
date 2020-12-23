#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Afzal Ansari"]
__all__ = ["AutoCorrelationTransformer", "PartialAutoCorrelationTransformer"]

import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf

from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class AutoCorrelationTransformer(_SeriesToSeriesTransformer):
    """
    Auto-correlation transformer.
    """

    _tags = {"univariate-only": True, "fit-in-transform": True}

    def __init__(
        self,
        adjusted=False,
        n_lags=None,
        qstat=False,
        fft=False,
        missing="none",
    ):
        self.adjusted = adjusted
        self.n_lags = n_lags
        self.qstat = qstat
        self.fft = fft
        self.missing = missing
        super(AutoCorrelationTransformer, self).__init__()

    def transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)

        # Passing an alpha values other than None would return confidence intervals
        # and break the signature of the series-to-series transformer
        zt = acf(
            z,
            adjusted=self.adjusted,
            nlags=self.n_lags,
            qstat=self.qstat,
            fft=self.fft,
            alpha=None,
            missing=self.missing,
        )
        return pd.Series(zt)


class PartialAutoCorrelationTransformer(_SeriesToSeriesTransformer):
    """
    Partial auto-correlation transformer.

    Parameters
    ----------
    n_lags : int
         largest lag for which pacf is returned
    method : str {'ywunbiased', 'ywmle', 'ols'}
         specifies which method for the calculations to use:
        - yw or ywunbiased : yule walker with bias correction in denominator
          for acovf. Default.
        - ywm or ywmle : yule walker without bias correction
        - ols - regression of time series on lags of it and on constant
        - ld or ldunbiased : Levinson-Durbin recursion with bias correction
        - ldb or ldbiased : Levinson-Durbin recursion without bias correction
    """

    _tags = {"univariate-only": True, "fit-in-transform": True}

    def __init__(
        self,
        n_lags=None,
        method="ywunbiased",
    ):
        self.n_lags = n_lags
        self.method = method
        super(PartialAutoCorrelationTransformer, self).__init__()

    def transform(self, Z, X=None):
        """
        pacf : 1d array
            partial autocorrelations, nlags elements, including lag zero
        confint : array, optional
            Confidence intervals for the PACF. Returned if confint is not None.
        """
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)

        # Passing an alpha values other than None would return confidence intervals
        # and break the signature of the series-to-series transformer
        zt = pacf(z, nlags=self.n_lags, method=self.method, alpha=None)
        return pd.Series(zt)
