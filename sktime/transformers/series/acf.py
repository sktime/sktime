#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# import numpy as np
# import pandas as pd

from statsmodels.tsa.stattools import acf
from sktime.transformers.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class AutoCorrelationFunctionTransformer(_SeriesToSeriesTransformer):
    _tags = {"transform-returns-same-time-index": True, "univariate-only": True}

    def __init__(
        self,
        unbiased=False,
        nlags=40,
        qstat=False,
        fft=False,
        alpha=None,
        missing="none",
    ):

        self.unbiased = unbiased
        self.nlags = nlags
        self.qstat = qstat
        self.fft = fft
        self.alpha = alpha
        self.missing = missing
        super(AutoCorrelationFunctionTransformer, self).__init__()

    def transform(self, Z, X=None):

        self.check_is_fitted()
        x = check_series(Z, enforce_univariate=True)
        zt = acf(
            x, self.unbiased, self.nlags, self.qstat, self.fft, self.alpha, self.missing
        )
        return zt
