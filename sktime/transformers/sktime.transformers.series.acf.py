# -*- coding: utf-8 -*-
# import numpy as np
# import pandas as pd

# from statsmodels.tsa.stattools import acf
from sktime.transformers.base import _SeriesToSeriesTransformer
from sktime.utils.validation.series import check_series


class AutoCorrelationFunctionTransformer(_SeriesToSeriesTransformer):
    def transform(
        self,
        x,
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

        self.check_is_fitted()
        x = check_series(x)
        return x
