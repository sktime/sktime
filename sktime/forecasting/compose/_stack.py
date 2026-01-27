#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements forecasters for combining forecasts via stacking."""

__author__ = ["mloning", "fkiraly", "indinewton"]
__all__ = ["StackingForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.split import SingleWindowSplitter
from sktime.utils.validation.forecasting import check_regressor
from sktime.utils.warnings import warn


class StackingForecaster(_HeterogenousEnsembleForecaster):
    """StackingForecaster.

    Stacks two or more Forecasters and uses a meta-model (regressor) to infer
    the final predictions from the predictions of the given forecasters.
    """

    _tags = {
        "authors": ["mloning", "fkiraly", "indinewton"],
        "capability:exogenous": True,
        "requires-fh-in-fit": True,
        "capability:missing_values": True,
        "capability:random_state": True,
        "property:randomness": "derandomized",
        "scitype:y": "univariate",
        "X-y-must-have-same-index": True,
    }

    def __init__(
        self,
        forecasters,
        regressor=None,
        random_state=None,
        n_jobs=None,
        passthrough=False,
    ):
        self.regressor = regressor
        self.random_state = random_state
        self.passthrough = passthrough

        super().__init__(forecasters=forecasters, n_jobs=n_jobs)

        self._anytagis_then_set("capability:exogenous", True, False, forecasters)
        self._anytagis_then_set("capability:missing_values", False, True, forecasters)
        self._anytagis_then_set("fit_is_empty", False, True, forecasters)

        # Force capability if passthrough is True
        if self.passthrough:
            self.set_tags(**{"capability:exogenous": True})

    def _fit(self, y, X, fh):
        forecasters = [x[1] for x in self.forecasters_]
        self.regressor_ = check_regressor(
            regressor=self.regressor, random_state=self.random_state
        )

        inner_fh = fh.to_relative(self.cutoff)
        cv = SingleWindowSplitter(fh=inner_fh)
        train_window, test_window = next(cv.split(y))
        y_train = y.iloc[train_window]
        y_test = y.iloc[test_window]
        
        if X is not None:
            X_test = X.iloc[test_window]
            X_train = X.iloc[train_window]
        else:
            X_test = None
            X_train = None

        self._fit_forecasters(forecasters, y_train, fh=inner_fh, X=X_train)
        y_preds = self._predict_forecasters(fh=inner_fh, X=X_test)

        y_meta = y_test.values
        X_meta = np.column_stack(y_preds)

        if self.passthrough and X_test is not None:
            X_test_np = X_test.values
            if X_test_np.ndim == 1:
                X_test_np = X_test_np.reshape(-1, 1)
            X_meta = np.column_stack([X_meta, X_test_np])

        self.regressor_.fit(X_meta, y_meta)
        self._fit_forecasters(forecasters, y, fh=fh, X=X)

        return self

    def _update(self, y, X=None, update_params=True):
        if update_params:
            warn("Updating `final regressor is not implemented", obj=self)
        for forecaster in self._get_forecaster_list():
            forecaster.update(y, X, update_params=update_params)
        return self

    def _predict(self, fh=None, X=None):
        y_preds = np.column_stack(self._predict_forecasters(fh=fh, X=X))

        if self.passthrough and X is not None:
            # --- CRITICAL FIX: Slice X to match the forecast horizon ---
            fh_abs = self.fh.to_absolute_index(self.cutoff)
            X_sliced = X.loc[fh_abs]
            # -----------------------------------------------------------
            
            X_np = X_sliced.values
            if X_np.ndim == 1:
                X_np = X_np.reshape(-1, 1)
            y_preds = np.column_stack([y_preds, X_np])

        y_pred = self.regressor_.predict(y_preds)
        index = self.fh.to_absolute_index(self.cutoff)
        return pd.Series(y_pred, index=index, name=self._y.name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sktime.forecasting.naive import NaiveForecaster
        f1 = NaiveForecaster()
        f2 = NaiveForecaster(strategy="mean", window_length=3)
        params = {"forecasters": [("f1", f1), ("f2", f2)]}
        return params