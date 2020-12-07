#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["BoxCoxTransformer"]

import numpy as np
import pandas as pd
from scipy.special import boxcox
from scipy.special import inv_boxcox

from sktime.transformers.base import _SeriesToSeriesTransformer
from sktime.utils.boxcox import boxcox_normmax
from sktime.utils.validation.series import check_series


class BoxCoxTransformer(_SeriesToSeriesTransformer):
    _tags = {"transform-returns-same-time-index": True, "univariate-only": True}

    def __init__(self, bounds=None, method="mle"):
        self.bounds = bounds
        self.method = method
        self.lambda_ = None
        super(BoxCoxTransformer, self).__init__()

    def fit(self, Z, X=None):
        z = check_series(Z, enforce_univariate=True)
        self.lambda_ = boxcox_normmax(z, bounds=self.bounds, method=self.method)
        self._is_fitted = True
        return self

    def transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        zt = boxcox(z.to_numpy(), self.lambda_)
        return pd.Series(zt, index=z.index)

    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        z = check_series(Z, enforce_univariate=True)
        zt = inv_boxcox(z.to_numpy(), self.lambda_)
        return pd.Series(zt, index=z.index)


class LogTransformer(_SeriesToSeriesTransformer):

    _tags = {"transform-returns-same-time-index": True}

    def transform(self, Z, X=None):
        self.check_is_fitted()
        Z = check_series(Z)
        return np.log(Z)

    def inverse_transform(self, Z, X=None):
        self.check_is_fitted()
        Z = check_series(Z)
        return np.exp(Z)
