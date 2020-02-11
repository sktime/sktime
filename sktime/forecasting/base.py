#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["BaseTemporalEstimator", "BaseForecaster", "RequiredForecastingHorizonMixin",
           "OptionalForecastingHorizonMixin", "DEFAULT_ALPHA"]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.exceptions import NotFittedError
from sktime.utils.validation.forecasting import check_cv
from sktime.utils.validation.forecasting import check_fh
from sktime.utils.validation.forecasting import check_y

DEFAULT_ALPHA = 0.05


class BaseTemporalEstimator(BaseEstimator):

    def __init__(self):
        self._oh = None  # observation horizon, i.e. time points seen in fit or update
        self._now = None  # time point in observation horizon now which to make forecasts
        self._is_fitted = False

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

    def _set_oh(self, y, update_now=True):
        """Set and update the observation horizon

        Parameters
        ----------
        y : pd.Series
        update_now : bool, optional (default=True)
            If True, sets `now` to the end of the observation horizon. Otherwise, leaves `now` unchanged.
        """
        # input checks
        oh = check_y(y)

        # for updating: append observation horizon to previous one
        if self.is_fitted:
            # update observation horizon, merging both series on time index,
            # overwriting old data with new data
            self._oh = oh.combine_first(self.oh)

        # for fitting: since no previous observation horizon is present, set new one
        else:
            self._oh = oh

        # by default, set now to the end of the observation horizon
        if update_now:
            self._set_now(oh.index[-1])

    @property
    def now(self):
        """Now, the time point at which to make forecasts.

        Returns
        -------
        now : int
        """
        return self._now

    def _set_now(self, now):
        """Set and update now, the time point at which to make forecasts.

        Parameters
        ----------
        now : int
        """
        if now not in self.oh.index:
            raise ValueError("Passed `now` value not in observation horizon")
        self._now = now


class BaseForecaster(BaseTemporalEstimator):
    _estimator_type = "forecaster"

    def __init__(self):
        self._fh = None
        super(BaseForecaster, self).__init__()

    def fit(self, y_train, fh=None, X_train=None):
        raise NotImplementedError()

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        raise NotImplementedError()

    def update(self, y_new, X_new=None, update_params=False):
        raise NotImplementedError()

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

        # input checks
        if X_test is not None or return_pred_int:
            raise NotImplementedError()

        self._check_is_fitted()
        if cv is None:
            # if no cv is provided, use default
            cv = SlidingWindowSplitter(fh=self.fh)
        else:
            # otherwise check provided cv
            cv = check_cv(cv)
            self._set_fh(cv.fh)

        # add the test set to the observation horizon, but keep `now` at the
        # end of the training set, so that we can make prediction iteratively
        # over the test set
        self._set_oh(y_test, update_now=False)

        # allocate lists for prediction results
        y_preds = []
        nows = []  # time points at which we predict

        # iteratively call update and predict, first update will contain only the
        # last window from the training set and no new data, so that we can make
        # the first prediction at the end of the training set
        for y_new, _ in self._iter(cv):
            # update: while the observation horizon is already updated, we still need to
            # update `now` and may want to update fitted parameters
            self.update(y_new, update_params=update_params)

            # predict: make a forecast at each step
            y_pred = self.predict(X=X_test, return_pred_int=return_pred_int, alpha=alpha)
            y_preds.append(y_pred)
            nows.append(self.now)

        # format results
        if len(self.fh) > 1:
            # return data frame when we predict multiple steps ahead
            y_preds = pd.DataFrame(y_preds).T
            y_preds.columns = nows
        else:
            # return series for single step ahead predictions
            y_preds = pd.concat(y_preds)

        return y_preds

    def update_predict_single(self, y_new, fh=None, X=None, update_params=False, return_pred_int=False,
                              alpha=DEFAULT_ALPHA):
        """Allows for more efficient update-predict routines than calling them sequentially"""
        # when nowcasting, X may be longer than y, X must be cut to same length as y so that same time points are
        # passed to update, the remaining time points of X are passed to predict
        if X is not None or return_pred_int:
            raise NotImplementedError()

        self.update(y_new, X_new=X, update_params=update_params)
        return self.predict(fh=fh, X=X, return_pred_int=return_pred_int, alpha=alpha)

    @property
    def fh(self):
        """The forecasting horizon"""
        # raise error if some method tries to accessed it before it has been set
        if self._fh is None:
            raise ValueError("No `fh` has been set yet.")
        return self._fh

    def compute_pred_errors(self, alpha=DEFAULT_ALPHA):
        """
        Prediction errors. If alpha is iterable, errors will be calculated for
        multiple confidence levels.
        """
        raise NotImplementedError()

    def compute_pred_int(self, y_pred, alpha=DEFAULT_ALPHA):
        """
        Get the prediction intervals for the forecast. If alpha is iterable, multiple
        intervals will be calculated.
        """
        errors = self.compute_pred_errors(alpha=alpha)

        # for multiple alphas, errors come in a list;
        # for single alpha, they come as a single pd.Series,
        # wrap it here into a list to make it iterable,
        # to avoid code duplication
        if isinstance(errors, pd.Series):
            errors = [errors]

        # compute prediction intervals
        pred_int = [
            pd.DataFrame({
                "lower": y_pred - error,
                "upper": y_pred + error
            })
            for error in errors
        ]

        # for a single alpha, return single pd.DataFrame
        if len(pred_int) == 1:
            return pred_int[0]

        # otherwise return list of pd.DataFrames
        return pred_int

    def score(self, y_test, fh=None, X=None):
        """
        Returns the SMAPE on the given forecast horizon.
        Parameters
        ----------
        y_test : pandas.Series
            Target time series to which to compare the forecasts.
        fh : int or array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.
        Returns
        -------
        score : float
            SMAPE score of self.predict(fh=fh, X=X) with respect to y.
        See Also
        --------
        :meth:`sktime.performance_metrics.forecasting.smape_loss`.`
        """
        # no input checks needed here, they will be performed
        # in predict and loss function
        return smape_loss(y_test, self.predict(fh=fh, X=X))

    def _iter(self, cv):
        """Iterate over the observation horizon starting at `now`

        Parameters
        ----------
        cv : cross-validation generator

        Yields
        ------
        y_train : pd.Series
            Training window of observation horizon
        y_test : pd.Series
            Test window of observation horizon
        """
        # get window length from cv generator
        window_length = cv.window_length

        # get starting point
        start = self.now - window_length + 1

        # check starting point for cv iterator is in observation horizon
        if start not in self.oh.index:
            raise ValueError(f"The `window_length`: {window_length} is longer than "
                             f"the available observation horizon `oh`.")

        # select subset to iterate over from observation horizon
        y = self.oh.iloc[start:]
        time_index = y.index

        # split subet of observation horizon into training and test set
        for training_window, test_window in cv.split(time_index):
            y_train = y.iloc[training_window]
            y_test = y.iloc[test_window]
            yield y_train, y_test

    def _set_fh(self, fh):
        """Check, set and update the forecasting horizon.

        Abstract base method, implemented by mixin classes.

        Parameters
        ----------
        fh : None, int, list, np.ndarray
        """
        #
        raise NotImplementedError()

    def _get_absolute_fh(self):
        """Convert the user-defined forecasting horizon relative to the end
        of the observation horizon into the absolute time index.

        Returns
        -------
        fh : numpy.ndarray
            The absolute time index of the forecasting horizon
        """
        # user defined forecasting horizon `fh` is relative to the end of the
        # observation horizon, i.e. `now`
        fh_abs = self.now + self.fh

        # for in-sample predictions, check if forecasting horizon is still within
        # observation horizon
        if any(fh_abs < 0):
            raise ValueError("Forecasting horizon `fh` includes time points "
                             "before observation horizon")
        return np.sort(fh_abs)

    def _get_index_fh(self):
        """Convert the step-ahead forecast horizon relative to the end
        of the observation horizon into the zero-based forecasting horizon
        for array indexing.
        Returns
        -------
        fh : numpy.ndarray
            The zero-based index of the forecasting horizon
        """
        return self.fh - 1


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
            # and overwrite with appropriate warning
            fh = check_fh(fh)
            if is_fitted:
                # raise warning if existing fh and new one don't match
                if self._fh is not None and not np.array_equal(fh, self._fh):
                    warn("The provided forecasting horizon `fh` is different from the "
                         "previously provided one; the new one will be used.")
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
