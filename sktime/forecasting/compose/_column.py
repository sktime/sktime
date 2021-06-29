#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = ["ColumnForecaster"]

import numpy as np
import pandas as pd
from sklearn import clone

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class ColumnForecaster(BaseForecaster):
    _tags = {
        "multivariate-only": True,
        "univariate-only": False,
        "requires-fh-in-fit": False,
    }

    _required_parameters = ["forecaster"]

    def __init__(self, forecaster):
        self.forecaster = forecaster
        super().__init__()

    def _fit(self, y, X=None, fh=None):

        n_columns = y.shape[1]
        self.forecasters_ = []

        for i in range(n_columns):
            forecaster = clone(self.forecaster)
            forecaster.fit(y.iloc[:, i], X=X, fh=fh)
            self.forecasters_.append(forecaster)

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):

        if return_pred_int:
            raise NotImplementedError()

        y_pred = np.zeros((len(self.fh), len(self.forecasters_)))
        for i, forecaster in enumerate(self.forecasters_):
            y_pred[:, i] = forecaster.predict(fh=fh, X=X)

        return pd.DataFrame(y_pred, index=self.fh.to_absolute(self.cutoff))
