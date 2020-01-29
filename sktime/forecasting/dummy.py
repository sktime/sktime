#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = ["DummyForecaster"]
__author__ = "Markus Löning"

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from sktime.forecasting.base import BaseForecasterOptionalFHinFit
from sktime.utils.validation.forecasting import validate_y, validate_cv


class DummyForecaster(BaseForecasterOptionalFHinFit):
    """
    Dummy forecaster for naive baseline forecasts
    """

    def __init__(self, strategy="last"):
        # input checks
        # allowed strategies an include: last, constant, seasonal-last, mean, median
        allowed_strategies = ("last",)
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy: {strategy}, expected one of {allowed_strategies}")

        self.strategy = strategy
        self.window_length = None
        self._last_window = None

        if self.strategy == "last":
            self.window_length = 1

        super(DummyForecaster, self).__init__()

    def fit(self, y, fh=None, X=None):

        # input checks
        # in-sample forecast
        if isinstance(fh, str) and fh == "insample":
            raise NotImplementedError

        # ignore exogenous variables X
        y = validate_y(y)
        self._set_fh(fh)

        # update observation horizon
        self._update_obs_horizon(y.index)
        self._now = self._obs_horizon[-1]

        self._last_window = y.iloc[-self.window_length:]
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):

        # input checks
        if isinstance(fh, str) and fh == "insample":
            raise NotImplementedError
        if return_conf_int:
            raise NotImplementedError

        check_is_fitted(self, "_is_fitted")
        self._set_fh(fh)

        # prediction
        return pd.Series(np.repeat(self._last_window.to_numpy(), len(self.fh)), index=self._now + self.fh)

    def update(self, y_new, X_new=None, update_params=False):

        # input checks
        check_is_fitted(self, "_is_fitted")
        y_new = validate_y(y_new)

        # update observation horizon
        self._obs_horizon = self._update_obs_horizon(y_new.index)
        self._now = self._obs_horizon[-1]

        # update naive predictions
        self._last_window = y_new.iloc[-self.window_length:]

        return self
