#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pandas as pd
from sklearn import clone

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class ColumnForecaster(BaseForecaster):
    _tags = {
        "multivariate-only": True,
        "univariate-only": False,
    }

    _required_parameters = ["forecaster"]

    def __init__(self, forecaster):
        self.forecaster = forecaster
        super().__init__()

    def _fit(self, y, X=None, fh=None):

        n_columns = X.shape[1]
        self.forecasters_ = []

        for i in range(n_columns):
            forecaster = clone(self.forecaster)
            forecaster.fit(X.iloc[:, i], X, fh=self.fh)
            self.forecasters_.append(forecaster)

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):

        y_pred = np.zeros((len(fh), len(self.forecasters_)))
        for i, forecaster in enumerate(self.forecasters_):
            y_pred[:, i] = forecaster.predict(self.fh, X=X)

        return pd.DataFrame(y_pred, index=self.fh.to_absolute(self.cutoff))
