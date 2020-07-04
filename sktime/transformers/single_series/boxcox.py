#!/usr/bin/env python3 -u
# coding: utf-8
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["BoxCoxTransformer"]

import pandas as pd
from scipy.special import boxcox
from scipy.special import inv_boxcox
from sktime.transformers.single_series.base import \
    BaseSingleSeriesTransformer
from sktime.utils.boxcox import boxcox_normmax
from sktime.utils.validation.forecasting import check_y


class BoxCoxTransformer(BaseSingleSeriesTransformer):

    def __init__(self, bounds=None, method="mle"):
        self.bounds = bounds
        self.method = method
        self.lambda_ = None
        super(BoxCoxTransformer, self).__init__()

    def fit(self, y_train, **fit_params):
        self.lambda_ = boxcox_normmax(y_train, bounds=self.bounds,
                                      method=self.method)
        self._is_fitted = True
        return self

    def transform(self, y, **transform_params):
        self.check_is_fitted()
        check_y(y)
        yt = boxcox(y.values, self.lambda_)
        return pd.Series(yt, index=y.index)

    def inverse_transform(self, y, **transform_params):
        self.check_is_fitted()
        check_y(y)
        yt = inv_boxcox(y.values, self.lambda_)
        return pd.Series(yt, index=y.index)

    def update(self, y_new, update_params=False):
        raise NotImplementedError()
