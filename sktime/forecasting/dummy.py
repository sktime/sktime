#!/usr/bin/env python3 -u
# coding: utf-8

__all__ = ["DummyForecaster"]
__author__ = "Markus Löning"

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from sktime.forecasting.base import BaseForecaster, ForecasterOptionalFHinFitMixin
from sktime.utils.validation.forecasting import validate_cv
from sktime.utils.validation.forecasting import validate_y


class DummyForecaster(BaseForecaster, ForecasterOptionalFHinFitMixin):
    """
    Dummy forecaster for naive baseline forecasts
    """

    def __init__(self, strategy="last"):
        # allowed strategies an include: last, constant, seasonal-last, mean, median
        allowed_strategies = ("last",)
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy: {strategy}, expected one of {allowed_strategies}")
        self.strategy = strategy
        super(DummyForecaster, self).__init__()

    def fit(self, y, fh=None, X=None):

        # in-sample forecast
        if isinstance(fh, str) and fh == "insample":
            raise NotImplementedError

        y = validate_y(y)
        self._validate_fh(fh, required=False, method="fit")
        self._update_obs_horizon(y.index)
        self._now = self._obs_horizon[-1]

        if self.strategy == "last":
            y_pred = y.iloc[-1]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._y_pred = y_pred
        self._is_fitted = True
        return self

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):
        if isinstance(fh, str) and fh == "insample":
            raise NotImplementedError

        self._check_is_fitted()
        self._validate_fh(fh, required=True, method="predict")

        return pd.Series(np.repeat(self._y_pred, len(self.fh)), index=self._now + self.fh)

    def update(self, y_new, X=None):
        check_is_fitted(self, "_is_fitted")
        y_new = validate_y(y_new)
        self._obs_horizon = self._update_obs_horizon(y_new.index)
        self._now = self._obs_horizon[-1]

        if self.strategy == "last":
            self._y_pred = y_new.iloc[-1]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return self

    def update_predict(self, y, fh=None, cv=None, X=None, return_conf_int=False, alpha=0.05):

        if return_conf_int:
            raise NotImplementedError

        if cv is None:
            return self.update(y, X=X).predict(fh=fh, return_conf_int=return_conf_int, alpha=alpha)

        y = validate_y(y)
        cv = validate_cv(cv)
        fh = cv.fh
        step_length = cv.step_length
        window_length = cv.window_length

        # create new index to make first prediction at end of training set
        index = self._obs_horizon[-window_length:].append(y.index).values

        # allocate lists for prediction results
        y_preds = []
        pred_timepoints = []  # time points at which we predict

        # iterative predict and update
        for i, (in_window, out_window) in enumerate(cv.split(index)):

            # first prediction from training set without updates
            if i == 0:
                y_pred = self.predict(fh)
                y_preds.append(y_pred)
                pred_timepoints.append(self._obs_horizon[-1])
                continue

            new_window = in_window[-step_length:]
            y_new = y.loc[new_window]
            self.update(y_new)

            y_pred = self.predict(fh)
            y_preds.append(y_pred)
            pred_timepoints.append(self._obs_horizon[-1])

        # concatenate predictions
        if len(fh) > 1:
            y_preds = pd.DataFrame(y_preds).T
            y_preds.columns = pred_timepoints
        else:
            y_preds = pd.concat(y_preds)
        return y_preds
