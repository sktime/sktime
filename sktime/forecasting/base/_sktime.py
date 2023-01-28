# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""sktime window forecaster base class."""

__author__ = ["@mloning", "@big-o", "fkiraly"]
__all__ = ["_BaseWindowForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA, BaseForecaster
from sktime.forecasting.model_selection import CutoffSplitter
from sktime.utils.datetime import _shift


class _BaseWindowForecaster(BaseForecaster):
    """Base class for forecasters that use sliding windows."""

    def __init__(self, window_length=None):
        super(_BaseWindowForecaster, self).__init__()
        self.window_length = window_length
        self.window_length_ = None

    def _predict(self, fh, X=None):
        """Predict core logic."""
        kwargs = {"X": X}

        # all values are out-of-sample
        if fh.is_all_out_of_sample(self.cutoff):
            y_pred = self._predict_fixed_cutoff(
                fh.to_out_of_sample(self.cutoff), **kwargs
            )

        # all values are in-sample
        elif fh.is_all_in_sample(self.cutoff):
            y_pred = self._predict_in_sample(fh.to_in_sample(self.cutoff), **kwargs)

        # both in-sample and out-of-sample values
        else:
            y_ins = self._predict_in_sample(fh.to_in_sample(self.cutoff), **kwargs)
            y_oos = self._predict_fixed_cutoff(
                fh.to_out_of_sample(self.cutoff), **kwargs
            )
            y_pred = pd.concat([y_ins, y_oos])

        # ensure pd.Series name attribute is preserved
        if isinstance(y_pred, pd.Series) and isinstance(self._y, pd.Series):
            y_pred.name = self._y.name

        return y_pred

    def _predict_fixed_cutoff(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Make single-step or multi-step fixed cutoff predictions.

        Parameters
        ----------
        fh : np.array
            all positive (> 0)
        X : pd.DataFrame
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series or pd.DataFrame
        """
        # assert all(fh > 0)
        y_pred = self._predict_last_window(
            fh, X, return_pred_int=return_pred_int, alpha=alpha
        )
        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            return y_pred
        else:
            index = fh.to_absolute(self.cutoff)
            return pd.Series(y_pred, index=index)

    def _predict_in_sample(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Make in-sample prediction using single-step moving-cutoff predictions.

        Parameters
        ----------
        fh : np.array
            all non-positive (<= 0)
        X : pd.DataFrame
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred : pd.DataFrame or pd.Series
        """
        if return_pred_int:
            raise NotImplementedError()

        y_train = self._y

        # generate cutoffs from forecasting horizon, note that cutoffs are
        # still based on integer indexes, so that they can be used with .iloc
        cutoffs = fh.to_relative(self.cutoff) + len(y_train) - 2
        cv = CutoffSplitter(cutoffs, fh=1, window_length=self.window_length_)
        return self._predict_moving_cutoff(y_train, cv, X, update_params=False)

    def _predict_last_window(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Predict core logic.

        Parameters
        ----------
        fh : np.array
        X : pd.DataFrame
        return_pred_int : bool
        alpha : float or list of floats

        Returns
        -------
        y_pred : np.array
        """
        raise NotImplementedError("abstract method")

    def _get_last_window(self):
        """Select last window."""
        # Get the start and end points of the last window.
        cutoff = self._cutoff
        start = _shift(cutoff, by=-self.window_length_ + 1)
        cutoff = cutoff[0]

        # Get the last window of the endogenous variable.
        y = self._y.loc[start:cutoff].to_numpy()

        # If X is given, also get the last window of the exogenous variables.
        X = self._X.loc[start:cutoff].to_numpy() if self._X is not None else None

        return y, X

    @staticmethod
    def _predict_nan(fh):
        """Predict nan if predictions are not possible."""
        return np.full(len(fh), np.nan)
