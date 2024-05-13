# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Sktime window forecaster base class."""

__author__ = ["mloning", "big-o", "fkiraly"]
__all__ = ["_BaseWindowForecaster"]

import numpy as np
import pandas as pd

from sktime.forecasting.base._base import BaseForecaster
from sktime.split import CutoffSplitter
from sktime.utils.datetime import _shift


class _BaseWindowForecaster(BaseForecaster):
    """Base class for forecasters that use sliding windows."""

    def __init__(self, window_length=None):
        super().__init__()
        self.window_length = window_length
        self.window_length_ = None

    def _predict_boilerplate(self, fh, **kwargs):
        """Dispatcher to in-sample and out-of-sample logic.

        In-sample logic is implemented in _predict_in_sample.
        Out-of-sample logic is implemented in _predict_fixed_cutoff.
        """
        cutoff = self._cutoff

        # all values are out-of-sample
        if fh.is_all_out_of_sample(cutoff):
            y_pred = self._predict_fixed_cutoff(fh.to_out_of_sample(cutoff), **kwargs)

        # all values are in-sample
        elif fh.is_all_in_sample(self.cutoff):
            y_pred = self._predict_in_sample(fh.to_in_sample(cutoff), **kwargs)

        # both in-sample and out-of-sample values
        else:
            y_ins = self._predict_in_sample(fh.to_in_sample(cutoff), **kwargs)
            y_oos = self._predict_fixed_cutoff(fh.to_out_of_sample(cutoff), **kwargs)

            if isinstance(y_ins, pd.DataFrame) and isinstance(y_oos, pd.Series):
                y_oos = y_oos.to_frame(y_ins.columns[0])

            y_pred = pd.concat([y_ins, y_oos])

        return y_pred

    def _predict(self, fh, X):
        """Predict core logic."""
        kwargs = {"X": X}

        y_pred = self._predict_boilerplate(fh, **kwargs)

        # ensure pd.Series name attribute is preserved
        if isinstance(y_pred, pd.Series) and isinstance(self._y, pd.Series):
            y_pred.name = self._y.name
        if isinstance(y_pred, pd.DataFrame) and isinstance(self._y, pd.Series):
            y_pred = y_pred.iloc[:, 0]
            y_pred.name = self._y.name

        return y_pred

    def _predict_fixed_cutoff(self, fh, X=None, **kwargs):
        """Make single-step or multi-step fixed cutoff predictions.

        Parameters
        ----------
        fh : np.array
            all positive (> 0)
        X : pd.DataFrame

        Returns
        -------
        y_pred = pd.Series or pd.DataFrame
        """
        # assert all(fh > 0)
        y_pred = self._predict_last_window(fh, X=X, **kwargs)
        if isinstance(y_pred, pd.Series) or isinstance(y_pred, pd.DataFrame):
            return y_pred
        else:
            index = fh.to_absolute_index(self.cutoff)
            return pd.Series(y_pred, index=index)

    def _predict_in_sample(self, fh, X=None, **kwargs):
        """Make in-sample prediction using single-step moving-cutoff predictions.

        Parameters
        ----------
        fh : np.array
            all non-positive (<= 0)
        X : pd.DataFrame

        Returns
        -------
        y_pred : pd.DataFrame or pd.Series
        """
        y_train = self._y

        # generate cutoffs from forecasting horizon, note that cutoffs are
        # still based on integer indexes, so that they can be used with .iloc
        cutoffs = fh.to_relative(self.cutoff) + len(y_train) - 2
        cv = CutoffSplitter(cutoffs, fh=1, window_length=self.window_length_)
        return self._predict_moving_cutoff(y_train, cv, X, update_params=False)

    def _predict_last_window(self, fh, X=None, **kwargs):
        """Predict core logic.

        Parameters
        ----------
        fh : np.array
        X : pd.DataFrame

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

    def _predict_nan(self, fh=None, method="predict", **kwargs):
        """Create a return DataFrame for predict-like method, with all np.nan entries.

        Parameters
        ----------
        fh : ForecastingHorizon of self, optional (default=None)
            retrieved from self.fh if None
        method : str, optional (default="predict")
            method name to generate return DataFrame for
            name of one of the BaseForecaster predict-like methods
        **kwargs : optional
            further kwargs to predict-like methods, e.g., coverage for predict_interval
            passed to self._get_columns

        Returns
        -------
        y_pred : pd.DataFrame
            return DataFrame
            index, columns are as expected
            all entries are np.nan
        """
        if fh is None:
            fh = self.fh

        index = fh.get_expected_pred_idx(y=self._y, cutoff=self.cutoff)
        columns = self._get_columns(method=method, **kwargs)

        y_pred = pd.DataFrame(np.nan, index=index, columns=columns)
        return y_pred
