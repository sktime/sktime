#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = "Markus Löning"

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from sktime.forecasting.base import ForecasterMixin
from sktime.utils.validation.forecasting import validate_cv
from sktime.utils.validation.forecasting import validate_fh
from sktime.utils.validation.forecasting import validate_y


class DummyForecaster(BaseEstimator, ForecasterMixin):
    """
    Dummy forecaster for naive forecasters approaches.

    Parameters
    ----------
    strategy : str{'mean', 'last', 'linear'}, optional (default='last')
        Naive forecasters strategy
    sp : int
        Seasonal periodicity
    """

    def __init__(self, strategy="last"):
        allowed_strategies = ("last",)
        if strategy not in allowed_strategies:
            raise ValueError(f"Unknown strategy: {strategy}, expected one of {allowed_strategies}")
        self.strategy = strategy
        super(DummyForecaster, self).__init__()

    def fit(self, y_train, fh=None):

        y_train = validate_y(y_train)
        self._time_index = y_train.index

        if fh is not None:
            self.fh = validate_fh(fh)

        if self.strategy == "last":
            y_pred = y_train.iloc[-1]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._y_pred = y_pred

        return self

    def predict(self, fh=None, return_conf_int=False, alpha=0.05):
        check_is_fitted(self, "_y_pred")
        # validate forecasting horizon
        # if no fh is passed to predict, check if it was passed to fit; if so, use it; otherwise raise error
        if fh is None and self.fh is None:
            raise ValueError("The forecasting horizon `fh` must be given to `fit` or `predict`, but found none.")

        # if fh is passed to predict, check if fh was also passed to fit; if so, check if they are the same; if
        # they are are not the same, raise warning and use the one passed to predict
        else:
            fh = validate_fh(fh)
            if self.fh is not None and not np.array_equal(fh, self.fh):
                warn("The forecasting horizon `fh` passed to `predict` is different "
                     "from the `fh` passed to `fit`, the one passed to predict will be used.")
            self.fh = fh  # use passed fh; overwrites fh if it was passed to fit already

        return pd.Series(np.repeat(self._y_pred, len(self.fh)), index=self._time_index[-1] + self.fh)

    def update(self, y_new):
        check_is_fitted(self, "_y_pred")
        y_new = validate_y(y_new)
        self._time_index = self._update_time_index(y_new.index)

        if self.strategy == "last":
            self._y_pred = y_new.iloc[-1]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._is_fitted = True
        return self

    def _check_is_fitted(self):
        check_is_fitted(self, "_is_fitted")

    def predict_update(self, y_test, cv, return_conf_int=False, alpha=0.05):

        if return_conf_int:
            raise NotImplementedError

        self._check_is_fitted()
        y_test = validate_y(y_test)
        cv = validate_cv(cv)
        fh = cv.fh
        step_length = cv.step_length
        window_length = cv.window_length

        # create new index to make first prediction at end of training set
        index = self._time_index[-window_length:].append(y_test.index).values

        # allocate lists for prediction results
        predictions = []
        prediction_timepoints = []  # time points at which we predict

        # iterative predict and update
        for i, (in_window, out_window) in enumerate(cv.split(index)):
            # first prediction from training set without updates
            if i == 0:
                y_pred = self.predict(fh)
                predictions.append(y_pred)
                prediction_timepoints.append(self._time_index[-1])
                continue

            new_window = in_window[-step_length:]
            y_new = y_test.loc[new_window]
            self.update(y_new)

            y_pred = self.predict(fh)
            predictions.append(y_pred)
            prediction_timepoints.append(self._time_index[-1])

        # concatenate predictions
        if len(fh) > 1:
            predictions = pd.DataFrame(predictions).T
            predictions.columns = prediction_timepoints
        else:
            predictions = pd.concat(predictions)
        return predictions
