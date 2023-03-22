# -*- coding: utf-8 -*-
"""Compositors that control stream and refitting behaviour of update."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import pandas as pd

from sktime.datatypes import ALL_TIME_SERIES_MTYPES
from sktime.datatypes._utilities import get_window
from sktime.forecasting.base._delegate import _DelegatedForecaster

# prepare tags to clone - exceptions are TAGS_TO_KEEP
TAGS_TO_KEEP = ["fit_is_empty", "X_inner_mtype", "y_inner_mtype"]
# fit must be executed to fit the wrapped estimator and remember the cutoff
# mtype tags are set so X/y is passed through, conversions happen in wrapped estimator
TAGS_TO_CLONE = _DelegatedForecaster().get_tags().keys()
TAGS_TO_CLONE = list(set(TAGS_TO_CLONE).difference(TAGS_TO_KEEP))


class UpdateRefitsEvery(_DelegatedForecaster):
    """Refits periodically when update is called.

    If update is called with update_params=True and refit_interval or more has
        elapsed since the last fit, refits the forecaster instead (call to fit).
            refitting is done on (potentially) all data seen so far.
        refit_window controls the lookback window on which refitting is done
            refit data is cutoff (inclusive) to cutoff minus refit_window (exclusive)

    Parameters
    ----------
    refit_interval : difference of sktime time indices (int or timedelta), optional
        interval that needs to elapse after which the first update defaults to fit
        default = 0, i.e., always refits, never updates
        if index of y seen in fit is integer or y is index-free container type,
            refit_interval must be int, and is interpreted as difference of int location
        if index of y seen in fit is timestamp, must be int or pd.Timedelta
            if pd.Timedelta, will be interpreted as time since last refit elapsed
            if int, will be interpreted as number of time stamps seen since last refit
    refit_window_size : difference of sktime time indices (int or timedelta), optional
        length of the data window to refit to in case update calls fit
        default = inf, i.e., refits to entire training data seen so far
    refit_window_lag : difference of sktime indices (int or timedelta), optional
        lag of the data window to refit to, w.r.t. cutoff, in case update calls fit
        default = 0, i.e., refit window ends with and includes cutoff
    """

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "forecaster_"

    _tags = {
        "fit_is_empty": False,
        "requires-fh-in-fit": False,
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
    }

    def __init__(
        self, forecaster, refit_interval=0, refit_window_size=None, refit_window_lag=0
    ):
        self.forecaster = forecaster
        self.forecaster_ = forecaster.clone()

        self.refit_interval = refit_interval
        self.refit_window_size = refit_window_size
        self.refit_window_lag = refit_window_lag

        super(UpdateRefitsEvery, self).__init__()

        self.clone_tags(forecaster, TAGS_TO_CLONE)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        # we need to remember the time we last fit, to compare to it in _update
        self.last_fit_cutoff_ = self.cutoff[0]
        estimator = self._get_delegate()
        estimator.fit(y=y, fh=fh, X=X)
        return self

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        estimator = self._get_delegate()
        time_since_last_fit = self.cutoff[0] - self.last_fit_cutoff_
        refit_interval = self.refit_interval
        refit_window_size = self.refit_window_size
        refit_window_lag = self.refit_window_lag

        _y = self._y
        _X = self._X

        # treat situation where indexing of y is in timedelta but differences are int
        #   in that case, interpret any integers as iloc index differences
        #   and replace integers with timedelta quantities before proceeding
        if isinstance(time_since_last_fit, pd.Timedelta):
            if isinstance(refit_window_lag, int):
                lag = min(refit_window_lag, len(_y))
                refit_window_lag = self.cutoff[0] - _y.index[-lag]
            if isinstance(refit_window_size, int):
                _y_lag = get_window(_y, lag=refit_window_lag)
                window_size = min(refit_window_size, len(_y_lag))
                refit_window_size = _y_lag.index[-window_size]
            if isinstance(refit_interval, int):
                index = min(refit_interval, len(_y))
                refit_interval = self.cutoff[0] - _y.index[-index]
        # case distinction based on whether the refit_interval period has elapsed
        #   if yes: call fit, on the specified window sub-set of all observed data
        if time_since_last_fit >= refit_interval and update_params:
            if refit_window_size is not None or refit_window_lag != 0:
                y_win = get_window(
                    _y, window_length=refit_window_size, lag=refit_window_lag
                )
                X_win = get_window(
                    _X, window_length=refit_window_size, lag=refit_window_lag
                )
            else:
                y_win = _y
                X_win = _X
            fh = self._fh
            estimator.fit(y=y_win, X=X_win, fh=fh, update_params=update_params)

            # remember that we just fitted the estimator
            self.last_fit_cutoff_ = self.cutoff[0]
        else:
            # if no: call update as usual
            estimator.update(y=y, X=X, update_params=update_params)
        return self

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.trend import TrendForecaster

        forecaster = TrendForecaster.create_test_instance()

        param1 = {"forecaster": forecaster}
        param2 = {"forecaster": forecaster, "refit_interval": 2, "refit_window_size": 3}

        return [param1, param2]


class UpdateEvery(_DelegatedForecaster):
    """Update only periodically when update is called.

    If update is called, behaves like update_params=False, unless update_interval has
        elapsed since the last "true" update = inner update with update_params=False.
        update_window controls the lookback window on which refitting is done
            refit data is cutoff (inclusive) to cutoff minus refit_window (exclusive)

    Caution: default value of update_interval means *no updates* after `fit`.

    Parameters
    ----------
    update_interval : difference of sktime time indices (int or timedelta), optional
        interval that needs to elapse until inner update call with update_params=True
        default = None = infinity, i.e., never updates
        if index of y seen in fit is integer or y is index-free container type,
            update_interval must be int, is interpreted as difference of int location
        if index of y seen in fit is timestamp, must be int or pd.Timedelta
            if pd.Timedelta, will be interpreted as time since last true update elapsed
            if int, will be interpreted as number of time stamps seen since last update
    """

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "forecaster_"

    _tags = {
        "fit_is_empty": False,
        "requires-fh-in-fit": False,
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
    }

    def __init__(self, forecaster, update_interval=None):
        self.forecaster = forecaster
        self.forecaster_ = forecaster.clone()

        self.update_interval = update_interval

        super(UpdateEvery, self).__init__()

        self.clone_tags(forecaster, TAGS_TO_KEEP)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        # we need to remember the time we last fit, to compare to it in _update
        self.last_update_cutoff_ = self.cutoff[0]
        estimator = self._get_delegate()
        estimator.fit(y=y, fh=fh, X=X)
        return self

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        estimator = self._get_delegate()
        time_since_last_update = self.cutoff[0] - self.last_update_cutoff_
        update_interval = self.update_interval

        _y = self._y

        # treat situation where indexing of y is in timedelta but differences are int
        #   in that case, interpret any integers as iloc index differences
        #   and replace integers with timedelta quantities before proceeding
        if isinstance(time_since_last_update, pd.Timedelta):
            if isinstance(update_interval, int):
                index = min(update_interval, len(_y))
                update_interval = self.cutoff[0] - _y.index[-index]
        # case distinction based on whether the update_interval period has elapsed
        # (None update_interval means infinite update_interval)
        #   if yes: call inner update with update_params=True, aka "true" update
        if update_interval is not None and time_since_last_update >= update_interval:
            estimator.update(y=y, X=X, update_params=update_params)

            # remember that we just updated the estimator
            self.last_update_cutoff_ = self.cutoff[0]
        else:
            # if no: call update, but with update_params=False
            estimator.update(y=y, X=X, update_params=False)
        return self

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.trend import TrendForecaster

        forecaster = TrendForecaster.create_test_instance()

        param1 = {"forecaster": forecaster}
        param2 = {"forecaster": forecaster, "update_interval": 2}

        return [param1, param2]


class DontUpdate(_DelegatedForecaster):
    """Turns off updates, i.e., ensures that forecaster is only fit and never updated.

    This is useful when comparing forecasters that update with forecasters that don't,
    in a set-up where all forecasters' `update` has `update_params=True` set.

    Shorthand for UpdateEvery with default values.

    Parameters
    ----------
    refit_interval : difference of sktime time indices (int or timedelta), optional
        interval that needs to elapse after which the first update defaults to fit
        default = 0, i.e., always refits, never updates
        if index of y seen in fit is integer or y is index-free container type,
            refit_interval must be int, and is interpreted as difference of int location
        if index of y seen in fit is timestamp, must be int or pd.Timedelta
            if pd.Timedelta, will be interpreted as time since last refit elapsed
            if int, will be interpreted as number of time stamps seen since last refit
    refit_window_size : difference of sktime time indices (int or timedelta), optional
        length of the data window to refit to in case update calls fit
        default = inf, i.e., refits to entire training data seen so far
    refit_window_lag : difference of sktime indices (int or timedelta), optional
        lag of the data window to refit to, w.r.t. cutoff, in case update calls fit
        default = 0, i.e., refit window ends with and includes cutoff
    """

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "forecaster_"

    _tags = {
        "fit_is_empty": False,
        "requires-fh-in-fit": False,
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
    }

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.forecaster_ = forecaster.clone()

        super(DontUpdate, self).__init__()

        self.clone_tags(forecaster, TAGS_TO_CLONE)

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        estimator = self._get_delegate()
        # we need to call this to ensure cutoff of estimator is updated
        estimator.update(y=y, X=X, update_params=False)
        return self

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.trend import TrendForecaster

        forecaster = TrendForecaster.create_test_instance()

        return {"forecaster": forecaster}
