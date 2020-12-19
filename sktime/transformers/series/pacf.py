#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# import numpy as np
# import pandas as pd

from statsmodels.tsa.stattools import pacf
from sktime.transformers.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class PartialAutoCorrelationFunctionTransformer(_SeriesToSeriesTransformer):
    _tags = {"transform-returns-same-time-index": True, "univariate-only": True}

    def __init__(
        self,
        unbiased=False,
        nlags=40,
        method="ywunbiased",
        alpha=None,
    ):
        """
        Partial autocorrelation estimated
        Parameters
        ----------
        x : 1d array
             observations of time series for which pacf is calculated
        nlags : int
             largest lag for which pacf is returned
        method : {'ywunbiased', 'ywmle', 'ols'}
             specifies which method for the calculations to use:

        - yw or ywunbiased : yule walker with bias correction in denominator
          for acovf. Default.
        - ywm or ywmle : yule walker without bias correction
        - ols - regression of time series on lags of it and on constant
        - ld or ldunbiased : Levinson-Durbin recursion with bias correction
        - ldb or ldbiased : Levinson-Durbin recursion without bias correction
          alpha : float, optional
        If a number is given, the confidence intervals for the given level are
        returned. For instance if alpha=.05, 95 % confidence intervals are
        returned where the standard deviation is computed according to
        1/sqrt(len(x))
        """

        self.unbiased = unbiased
        self.nlags = nlags
        self.method = method
        self.alpha = alpha
        super(PartialAutoCorrelationFunctionTransformer, self).__init__()

    def transform(self, Z, X=None):

        """
        pacf : 1d array
            partial autocorrelations, nlags elements, including lag zero
        confint : array, optional
            Confidence intervals for the PACF. Returned if confint is not None.
            )
        """
        self.check_is_fitted()
        x = check_series(Z, enforce_univariate=True)
        zt = pacf(x, self.unbiased, self.nlags, self.method, self.alpha)

        return zt
