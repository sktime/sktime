#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = ["DummyForecaster"]
__author__ = "Markus Löning"

from warnings import warn

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecasterOptionalFHinFit
from sktime.utils.validation.forecasting import validate_y


class DummyForecaster(BaseForecasterOptionalFHinFit):
    """
    Dummy forecaster for naive baseline forecasts
    """

    def __init__(self, strategy="last", window_length=None):
        # input checks
        # allowed strategies an include: last, constant, seasonal-last, mean, median
        allowed_strategies = ("last", "mean")
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy: {strategy}; expected one of {allowed_strategies}")
        self.strategy = strategy

        if self.strategy == "last":
            if window_length is not None:
                warn("For the `last` strategy the `window_length` value will be ignored.")
            self.window_length = 1

        if self.strategy == "mean":
            if window_length is None:
                raise ValueError("`window_length` has to be specified "
                                 "when the `mean` strategy is used.")
            if window_length < 2:
                raise ValueError("`window_length` must be > 2; for `window_length`=1, "
                                 "use the `last` strategy.")
            self.window_length = window_length

        self._last_window = None

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
        self._set_obs_horizon(y.index)

        if self.window_length > len(self._obs_horizon):
            raise ValueError(f"The window length: {self.window_length} is larger than "
                             f"the training series.")

        self._last_window = y.iloc[-self.window_length:]
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=BaseForecasterOptionalFHinFit._DEFAULT_ALPHA):

        # input checks
        self._check_is_fitted()

        if isinstance(fh, str) and fh == "insample":
            raise NotImplementedError
        if return_conf_int:
            raise NotImplementedError

        # set fh
        self._set_fh(fh)

        # compute prediction
        if self.strategy == "last":
            y_pred = self._last_window.iloc[-1]

        if self.strategy == "mean":
            y_pred = self._last_window.mean()

        # return as series with correct time index
        return pd.Series(np.repeat(y_pred, len(self.fh)), index=self._now + self.fh)

    def update(self, y_new, X_new=None, update_params=False):

        # input checks
        self._check_is_fitted()

        y_new = validate_y(y_new)

        # update observation horizon
        self._set_obs_horizon(y_new.index)

        # update last window
        self._last_window = y_new.iloc[-self.window_length:]

        return self
