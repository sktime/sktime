__all__ = ["BaseForecaster", "ForecasterOptionalFHinFitMixin", "BaseForecasterRequiredFHinFitMixin"]
__author__ = ["Markus Löning"]

from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator

from sktime.utils.validation.forecasting import validate_fh
from sktime.utils.validation.forecasting import validate_obs_horizon


class BaseForecaster(BaseEstimator):
    """
    Base class for forecasters.
    """
    _estimator_type = "forecaster"

    def __init__(self):
        self._obs_horizon = None  # keep track of observation horizon of target series
        self._now = None  # keep track of point in observation horizon at which to predict
        self.is_fitted = False
        self._fh = None

    def fit(self, y, fh=None, X=None):
        raise NotImplementedError

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):
        raise NotImplementedError

    def update(self, y, X=None, update_params=False):
        raise NotImplementedError

    def update_predict(self, y, fh=None, cv=None, X=None, update_params=False, return_conf_int=False, alpha=0.05):
        raise NotImplementedError

    def score(self, y, fh=None, X=None):
        raise NotImplementedError

    def score_update(self, y, cv=None, X=None, update_params=False):
        raise NotImplementedError

    def _update_obs_horizon(self, obs_horizon):
        """
        Update observation horizon
        """
        obs_horizon = validate_obs_horizon(obs_horizon)

        # for fitting when no previous observation horizon is present, set new observation horizon
        if self._obs_horizon is None:
            new_obs_horizon = obs_horizon

        # for updating, append observation horizon to previous one
        else:
            new_obs_horizon = self._obs_horizon.append(obs_horizon)
            if not new_obs_horizon.is_monotonic:
                raise ValueError("Updated time index is no longer monotonically increasing. Data passed "
                                 "to `update` must contain more recent data than data passed to `fit`.")

        # update observation horizon
        self._obs_horizon = new_obs_horizon
        return new_obs_horizon

    @property
    def fh(self):
        """Protect the forecasting horizon"""
        return self._fh


class ForecasterOptionalFHinFitMixin:
    """Base class for forecasters which can take the forecasting horizon either during fitting or prediction."""

    def _validate_fh(self, fh):
        """Validate and set forecasting horizon"""

        # check if fitted, fh can only be set if not fitted already
        is_fitted = self._is_fitted if hasattr(self, "_is_fitted") else False

        if fh is None:
            if is_fitted:
                # if no fh passed and there is none already, raise error
                if self.fh is None:
                    raise ValueError("The forecasting horizon `fh` must be passed either to `fit` or `predict`, "
                                     "but was found in neither.")
                # otherwise if no fh passed, but there is one already, we simply use that one
        else:
            # if fh is passed, validate first, then check if there is one already,
            # and overwrite with appropriate warning
            fh = validate_fh(fh)
            if is_fitted:
                # raise warning if existing fh and new one don't match
                if self.fh is not None and not np.array_equal(fh, self.fh):
                    warn("The provided forecasting horizon `fh` is different from the previous one; "
                         "the new one will be used.")
            self._fh = fh


class BaseForecasterRequiredFHinFitMixin:
    """Base class for forecasters which require the forecasting horizon during fitting."""

    def _validate_fh(self, fh):
        is_fitted = self._is_fitted if hasattr(self, "_is_fitted") else False

        if fh is None:
            if is_fitted:
                # intended workflow, no fh is passed when the forecaster is already fitted
                pass
            else:
                # fh must be passed when forecaster is not fitted yet
                raise ValueError("The forecasting horizon `fh` must be passed to `fit`, "
                                 "but none was found.")
        else:
            fh = validate_fh(fh)
            if is_fitted:
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
