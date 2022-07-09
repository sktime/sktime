# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""sktime window forecaster base class."""

__author__ = ["@mloning", "@big-o"]
__all__ = ["_BaseWindowForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA, BaseForecaster
from sktime.forecasting.model_selection import CutoffSplitter, SlidingWindowSplitter
from sktime.utils.datetime import _shift
from sktime.utils.validation.forecasting import check_cv


class _BaseWindowForecaster(BaseForecaster):
    """Base class for forecasters that use sliding windows."""

    def __init__(self, window_length=None):
        super(_BaseWindowForecaster, self).__init__()
        self.window_length = window_length
        self.window_length_ = None

    def update_predict(
        self,
        y,
        cv=None,
        X=None,
        update_params=True,
    ):
        """Make and update predictions iteratively over the test set.

        Parameters
        ----------
        y : pd.Series
        cv : temporal cross-validation generator, optional (default=None)
        X : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=True)

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
        """
        if cv is not None:
            cv = check_cv(cv)
        else:
            cv = SlidingWindowSplitter(
                self.fh.to_relative(self.cutoff),
                window_length=self.window_length_,
                start_with_window=False,
            )
        return self._predict_moving_cutoff(y, cv, X, update_params=update_params)

    def _predict(self, fh, X=None):
        """Predict core logic."""
        kwargs = {"X": X}

        # all values are out-of-sample
        if fh.is_all_out_of_sample(self.cutoff):
            return self._predict_fixed_cutoff(
                fh.to_out_of_sample(self.cutoff), **kwargs
            )

        # all values are in-sample
        elif fh.is_all_in_sample(self.cutoff):
            return self._predict_in_sample(fh.to_in_sample(self.cutoff), **kwargs)

        # both in-sample and out-of-sample values
        else:
            y_ins = self._predict_in_sample(fh.to_in_sample(self.cutoff), **kwargs)
            y_oos = self._predict_fixed_cutoff(
                fh.to_out_of_sample(self.cutoff), **kwargs
            )
            return pd.concat([y_ins, y_oos])

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

    def _update_predict_single(self, y, fh, X=None, update_params=True):
        """Update and make forecasts, core logic..

        Implements default behaviour of calling update and predict
        sequentially, but can be overwritten by subclasses
        to implement more efficient updating algorithms when available.

        Parameters
        ----------
        y
        fh
        X
        update_params

        Returns
        -------
        predictions
        """
        self.update(y=y, X=X, update_params=update_params)
        return self._predict(fh=fh, X=X)
