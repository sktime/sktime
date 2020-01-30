__all__ = ["_BaseForecaster", "_BaseForecasterOptionalFHinFit", "_BaseForecasterRequiredFHinFit"]
__author__ = ["Markus Löning"]

from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from sktime.utils.validation.forecasting import validate_cv
from sktime.utils.validation.forecasting import validate_fh
from sktime.utils.validation.forecasting import validate_time_index
from sktime.utils.validation.forecasting import validate_y


class _BaseForecaster(BaseEstimator):
    """
    Base class for forecasters.
    """
    _estimator_type = "forecaster"

    def __init__(self):
        self._obs_horizon = None  # observation horizon, i.e. time points seen in fit or update
        self._now = None  # time point in observation horizon from which to make forecasts
        self._is_fitted = False
        self._fh = None  # forecasting horizon, i.e. time points to forecast, relative to now

    def fit(self, y_train, fh=None, X_train=None):
        """Fit model to training data"""
        raise NotImplementedError

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):
        """Forecast"""
        raise NotImplementedError

    def update(self, y_new, X_new=None, update_params=False):
        """Update model, including observation horizon used to make predictions and/or model parameters"""
        raise NotImplementedError

    def update_predict(self, y_test, cv, X_test=None, update_params=False, return_conf_int=False, alpha=0.05):
        """Model evaluation with temporal cross-validation"""
        # temporal cross-validation is performed for model evaluation, returning
        # predictions for all time points of the new time series y (i.e. y_test)

        # input checks
        # when nowcasting, X may be longer than y, X must be cut to same length as y so that same time points are
        # passed to update, the remaining time points of X are passed to predict
        if X_test is not None:
            raise NotImplementedError

        if return_conf_int:
            raise NotImplementedError

        # input checks
        y_test = validate_y(y_test)
        cv = validate_cv(cv)

        # check forecasting horizon
        fh = cv.fh
        self._set_fh(fh)

        # allocate lists for prediction results
        y_preds = []
        pred_timepoints = []  # time points at which we predict

        # first prediction from training set without updates
        y_pred = self.predict()
        y_preds.append(y_pred)
        pred_timepoints.append(self._now)

        # iterative predict and update
        for new in self._iter(y_test, cv):
            # update
            self.update(y_test[new], update_params=update_params)

            # predict
            y_pred = self.predict()
            y_preds.append(y_pred)
            pred_timepoints.append(self._now)

        # after evaluation, reset to fitted
        # self._reset_to_fitted()

        # format predictions
        if len(self.fh) > 1:
            # return data frame when we predict multiple steps ahead
            y_preds = pd.DataFrame(y_preds).T
            y_preds.columns = pred_timepoints
        else:
            # return series for single step ahead predictions
            y_preds = pd.concat(y_preds)

        return y_preds

    def update_predict_single(self, y_new, fh=None, X=None, update_params=False, return_conf_int=False, alpha=0.05):
        """Allows for more efficient update-predict routines than calling them sequentially"""
        # when nowcasting, X may be longer than y, X must be cut to same length as y so that same time points are
        # passed to update, the remaining time points of X are passed to predict
        if X is not None:
            raise NotImplementedError

        self.update(y_new, X_new=X, update_params=update_params)
        return self.predict(fh=fh, X=X, return_conf_int=return_conf_int, alpha=alpha)

    def score(self, y_test, fh=None, X=None):
        y_pred = self.predict(fh=fh, X=X)
        # compute scores against y_test
        raise NotImplementedError

    def update_score(self, y_test, cv=None, X=None, update_params=False):
        """Model evaluation with temporal cross-validation"""
        y_pred = self.update_predict(y_test, cv=cv, X_test=X, update_params=update_params)
        # compute scores
        raise NotImplementedError

    @property
    def fh(self):
        """Protect the forecasting horizon"""
        return self._fh

    @property
    def is_fitted(self):
        return self._is_fitted

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise NotFittedError(f"This instance of {self.__class__.__name__} has not "
                                 f"been fitted yet; please call `fit` first.")

    @property
    def now(self):
        return self._now

    def _set_obs_horizon(self, obs_horizon, update_now=True):
        """
        Update observation horizon
        """
        obs_horizon = validate_time_index(obs_horizon)

        # for fitting: since no previous observation horizon is present, set new one
        if not self.is_fitted:
            new_obs_horizon = obs_horizon

        # for updating: append observation horizon to previous one
        else:
            new_obs_horizon = self._obs_horizon.append(obs_horizon)
            if not new_obs_horizon.is_monotonic:
                raise ValueError("Updated time index is no longer monotonically increasing. Data passed "
                                 "to `update` must contain more recent observations than data previously "
                                 "passed to `fit` or `update`.")

        # update observation horizon
        self._obs_horizon = new_obs_horizon

        # by default, update now when new obs horizon is updated
        if update_now:
            self._set_now()

    def _set_now(self, now=None):
        if now is None:
            if self.fh == "insample":
                # now = self._first_window[-1]
                raise NotImplementedError
            else:
                now = self._obs_horizon[-1]
        else:
            if now not in self._obs_horizon:
                raise ValueError("Passed value not in current observation horizon")

        self._now = now

    def _set_fh(self, fh):
        raise NotImplementedError

    def _reset_to_fitted(self):
        """Reset model to fitted state after running model evaluation"""
        raise NotImplementedError

    def _iter(self, y, cv):
        # set up temporal cv
        window_length = cv.window_length
        step_length = cv.step_length

        # check consistent length, window cannot go further back in obs horizon
        # than length of obs horizon
        if window_length > len(self._obs_horizon):
            raise ValueError(f"The window length: {window_length} is larger than "
                             f"the current observation horizon: {len(self._obs_horizon)}")

        # combine obs horizons
        new_obs_horizon = self._obs_horizon.append(y.index)

        # time index to use for updating and predicting
        time_index = new_obs_horizon[-(len(y) + window_length - step_length):]

        # temporal cv
        for i, _ in cv.split(time_index):
            # not all observations in the window will be new to the forecaster,
            # only the last ones, depending on the step length
            yield i[-step_length:]  # return only index of new data points


class _BaseForecasterOptionalFHinFit(_BaseForecaster):
    """Base class for forecasters which can take the forecasting horizon either during fitting or prediction."""

    def _set_fh(self, fh):
        """Validate and set forecasting horizon"""

        if fh is None:
            if self.is_fitted:
                # if no fh passed and there is none already, raise error
                if self.fh is None:
                    raise ValueError("The forecasting horizon `fh` must be passed either to `fit` or `predict`, "
                                     "but was found in neither.")
                # otherwise if no fh passed, but there is one already, we simply use that one
        else:
            # if fh is passed, validate first, then check if there is one already,
            # and overwrite with appropriate warning
            fh = validate_fh(fh)
            if self.is_fitted:
                # raise warning if existing fh and new one don't match
                if self.fh is not None and not np.array_equal(fh, self.fh):
                    warn("The provided forecasting horizon `fh` is different from the previous one; "
                         "the new one will be used.")
            self._fh = fh


class _BaseForecasterRequiredFHinFit(_BaseForecaster):
    """Base class for forecasters which require the forecasting horizon during fitting."""

    def _set_fh(self, fh):

        if fh is None:
            if self.is_fitted:
                # intended workflow, no fh is passed when the forecaster is already fitted
                pass
            else:
                # fh must be passed when forecaster is not fitted yet
                raise ValueError("The forecasting horizon `fh` must be passed to `fit`, "
                                 "but none was found.")
        else:
            fh = validate_fh(fh)
            if self.is_fitted:
                if not np.array_equal(fh, self.fh):
                    # raise error if existing fh and new one don't match
                    raise ValueError(
                        f"A different forecasting horizon `fh` has been provided from the one seen in `fit`. "
                        f"Training of {self.__class__.__name__} depends on the forecasting horizon. "
                        f"If you want to change the forecasting horizon, please re-fit the forecaster.")
                # if existing one and new match, ignore new one
                pass
            else:
                # intended workflow: fh is passed when not forecaster is not fitted yet
                self._fh = fh
