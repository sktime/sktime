#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["BaseSktimeForecaster", "BaseLastWindowForecaster"]

from contextlib import contextmanager

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base import DEFAULT_ALPHA
from sktime.forecasting.model_selection import SlidingWindowSplitter, ManualWindowSplitter
from sktime.utils.exceptions import NotFittedError
from sktime.utils.validation.forecasting import check_y, check_cv, check_fh


class BaseSktimeForecaster(BaseForecaster):

    def __init__(self):
        self._oh = None  # observation horizon, i.e. time points seen in fit or update
        self._cutoff = None  # time point in observation horizon cutoff which to make forecasts
        self._is_fitted = False
        self._fh = None
        super(BaseSktimeForecaster, self).__init__()

    @property
    def is_fitted(self):
        """Has `fit` been called?"""
        return self._is_fitted

    def _check_is_fitted(self):
        """Check if the forecaster has been fitted.

        Raises
        ------
        NotFittedError
            if the forecaster has not been fitted yet.
        """
        if not self.is_fitted:
            raise NotFittedError(f"This instance of {self.__class__.__name__} has not "
                                 f"been fitted yet; please call `fit` first.")

    @property
    def oh(self):
        """The observation horizon, i.e. the seen data
        passed either to `fit` or one of the `update` methods.

        Returns
        -------
        oh : pd.Series
            The available observation horizon
        """
        return self._oh

    def _set_oh(self, y):
        """Set and update the observation horizon

        Parameters
        ----------
        y : pd.Series
        """
        # input checks
        oh = check_y(y, allow_empty=True)

        # for updating: append observation horizon to previous one
        if self.is_fitted:
            # update observation horizon, merging both series on time index,
            # overwriting old data with new data
            self._oh = oh.combine_first(self.oh)

        # for fitting: since no previous observation horizon is present, set new one
        else:
            self._oh = oh

        # by default, set cutoff to the end of the observation horizon
        self._set_cutoff(oh.index[-1])

    @property
    def cutoff(self):
        """Now, the time point at which to make forecasts.

        Returns
        -------
        cutoff : int
        """
        return self._cutoff

    def _set_cutoff(self, cutoff):
        """Set and update cutoff, the time point at which to make forecasts.

        Parameters
        ----------
        cutoff : int
        """
        self._cutoff = cutoff

    @contextmanager
    def _detached_cutoff(self):
        """context manager to detach cutoff"""
        cutoff = self.cutoff  # remember initial cutoff
        try:
            yield
        finally:
            # re-set cutoff to initial state
            self._set_cutoff(cutoff)

    @property
    def fh(self):
        """The forecasting horizon"""
        # raise error if some method tries to accessed it before it has been set
        if self._fh is None:
            raise ValueError("No `fh` has been set yet.")
        return self._fh

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Abstract base method, implemented by mixin classes.

        Parameters
        ----------
        fh : None, int, list, np.array
        """
        #
        raise NotImplementedError()

    def _get_absolute_fh(self, fh=None):
        """Convert the user-defined forecasting horizon relative to the end
        of the observation horizon into the absolute time index.

        Returns
        -------
        fh : np.array
            The absolute time index of the forecasting horizon
        """
        # user defined forecasting horizon `fh` is relative to the end of the
        # observation horizon, i.e. `cutoff`
        if fh is None:
            fh = self.fh
        fh_abs = self.cutoff + fh

        # for in-sample predictions, check if forecasting horizon is still within
        # observation horizon
        if any(fh_abs < 0):
            raise ValueError("Forecasting horizon `fh` includes time points "
                             "before observation horizon")
        return np.sort(fh_abs)

    def _get_array_index_fh(self, fh=None):
        """Convert the step-ahead forecast horizon relative to the end
        of the observation horizon into the zero-based forecasting horizon
        for array indexing.
        Returns
        -------
        fh : np.array
            The zero-based index of the forecasting horizon
        """
        if fh is None:
            fh = self.fh
        return fh - 1

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Predict

        Parameters
        ----------
        fh : int, list or np.array
        X : pd.DataFrame
        return_pred_int : bool, optional (default=False)
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
        y_pred_int : pd.DataFrame
        """
        self._check_is_fitted()
        self._set_fh(fh)
        return self._predict(self.fh, X=X, return_pred_int=return_pred_int, alpha=alpha)

    def update(self, y_new, X_new=None, update_params=False):
        self._check_is_fitted()
        if update_params:
            raise NotImplementedError()
        y_new = check_y(y_new, allow_empty=True)
        self._set_oh(y_new)
        return self

    def update_predict(self, y_test, cv=None, X_test=None, update_params=False, return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
        """Make predictions and updates iteratively over the test set.

        Parameters
        ----------
        y_test : pd.Series
        cv : cross-validation generator, optional (default=None)
        X_test : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=False)
        return_pred_int : bool, optional (default=False)
        alpha : int or list of ints, optional (default=None)

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
        """
        if return_pred_int:
            raise NotImplementedError()
        cv = check_cv(cv) if cv is not None else SlidingWindowSplitter(fh=self.fh)
        return self._predict_moving_cutoff(y_test, cv, X=X_test, update_params=update_params,
                                           return_pred_int=return_pred_int, alpha=alpha)

    def _predict_moving_cutoff(self, y, cv, X=None, update_params=False, return_pred_int=False,
                               alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()
        fh = cv.fh
        y_preds = []
        with self._detached_cutoff():
            for new_window, _ in enumerate(cv.split(y)):
                y_new = y.iloc[new_window]
                self.update(y_new, update_params=update_params)
                y_pred = self._predict(fh, X=X, return_pred_int=return_pred_int, alpha=alpha)
                y_preds.append(y_pred)
        cutoffs = cv.get_cutoffs(y)
        return _format_moving_cutoff_predictions(y_preds, cutoffs)

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        raise NotImplementedError("abstract method")


class OptionalForecastingHorizonMixin:
    """Mixin class for forecasters which can take the forecasting horizon either
    during fitting or prediction."""

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Parameters
        ----------
        fh : None, int, list or np.ndarray
        """
        if hasattr(self, "is_fitted"):
            is_fitted = self.is_fitted
        else:
            raise AttributeError("No `is_fitted` attribute found")

        if fh is None:
            if is_fitted:
                # if no fh passed and there is none already, raise error
                if self._fh is None:
                    raise ValueError("The forecasting horizon `fh` must be passed either to `fit` or `predict`, "
                                     "but was found in neither.")
                # otherwise if no fh passed, but there is one already, we can simply use that one
        else:
            # if fh is passed, validate first, then check if there is one already,
            # and overwrite

            # a warning should only be raised if fh passed to fit is overwritten, but no warning is required
            # when no fh has been provided in fit, and different fhs are passed to predict, but this requires
            # to keep track of whether fh has been passed to fit or not, hence not implemented for cutoff
            fh = check_fh(fh)
            self._fh = fh


class RequiredForecastingHorizonMixin:
    """Mixin class for forecasters which require the forecasting horizon during fitting."""

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Parameters
        ----------
        fh : None, int, list, np.ndarray
        """

        msg = f"This is because fitting of the `{self.__class__.__name__}` depends on `fh`. "

        if hasattr(self, "is_fitted"):
            is_fitted = self.is_fitted
        else:
            raise AttributeError("No `is_fitted` attribute found")

        if fh is None:
            if is_fitted:
                # intended workflow, no fh is passed when the forecaster is already fitted
                pass
            else:
                # fh must be passed when forecaster is not fitted yet
                raise ValueError("The forecasting horizon `fh` must be passed to `fit`, "
                                 "but none was found. " + msg)
        else:
            fh = check_fh(fh)
            if is_fitted:
                if not np.array_equal(fh, self._fh):
                    # raise error if existing fh and new one don't match
                    raise ValueError(
                        f"A different forecasting horizon `fh` has been provided from "
                        f"the one seen in `fit`. If you want to change the forecasting "
                        f"horizon, please re-fit the forecaster. " + msg)
                # if existing one and new match, ignore new one
                pass
            else:
                # intended workflow: fh is passed when forecaster is not fitted yet
                self._fh = fh


class BaseLastWindowForecaster(BaseSktimeForecaster):

    def __init__(self):
        super(BaseLastWindowForecaster, self).__init__()
        self._window_length = None

    def update_predict(self, y_test, cv=None, X_test=None, update_params=False, return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
        cv = check_cv(cv) if cv is not None else SlidingWindowSplitter(self.fh, window_length=self.window_length)
        return self._predict_moving_cutoff(y_test, cv, X=X_test, update_params=update_params,
                                           return_pred_int=return_pred_int, alpha=alpha)

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()

        is_oos = fh > 0
        is_ins = np.logical_not(is_oos)

        fh_oos = fh[is_oos]
        fh_ins = fh[is_ins]

        if all(is_oos):
            return self._predict_fixed_cutoff(fh_oos, X=X, return_pred_int=return_pred_int, alpha=alpha)
        elif all(is_ins):
            return self._predict_in_sample(fh_ins, X=X, return_pred_int=return_pred_int, alpha=alpha)
        else:
            y_ins = self._predict_in_sample(fh_ins, X=X, return_pred_int=return_pred_int, alpha=alpha)
            y_oos = self._predict_fixed_cutoff(fh_oos, X=X, return_pred_int=return_pred_int, alpha=alpha)
            return y_ins.append(y_oos)

    def _predict_fixed_cutoff(self, fh, X=None, return_pred_int=False, alpha=None):
        # assert all(fh > 0)
        y_pred = self._predict_last_window(fh, X=X, return_pred_int=return_pred_int, alpha=alpha)
        index = self._get_absolute_fh(fh)
        return pd.Series(y_pred, index=index)

    def _predict_in_sample(self, fh, X=None, return_pred_int=False, alpha=None):
        # assert all(fh <= 0)
        cutoffs = self.cutoff + fh - 1
        cv = ManualWindowSplitter(cutoffs, fh=1, window_length=self.window_length)
        y_train = self.oh
        return self._predict_moving_cutoff(y_train, cv, X=X, update_params=False, return_pred_int=return_pred_int,
                                           alpha=alpha)

    def _predict_last_window(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        raise NotImplementedError("abstract method")

    def _get_last_window(self):
        start = self.cutoff - self.window_length + 1
        end = self.cutoff
        return self.oh.loc[start:end].values

    @property
    def window_length(self):
        return self._window_length


def _format_moving_cutoff_predictions(y_preds, cutoffs):
    """Helper function to format moving-cutoff predictions"""
    if len(y_preds) == 1:
        # return series for single step ahead predictions
        y_pred = pd.concat(y_preds)
    else:
        # return data frame when we predict multiple steps ahead
        y_pred = pd.DataFrame(y_preds).T
        y_pred.columns = cutoffs

    return y_pred
