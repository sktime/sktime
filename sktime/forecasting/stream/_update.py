# -*- coding: utf-8 -*-
"""Compositors that control stream and refitting behaviour of update."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from sklearn import clone

from sktime.datatypes._utilities import get_window
from sktime.forecasting.base._delegate import _DelegatedForecaster


class UpdateRefitsEvery(_DelegatedForecaster):
    """Refits periodically when update is called.

    If update is called with update_params=True and refit_interval or more has
        elapsed since the last fit, refits the forecaster instead (call to fit).
        refit_window controls the lookback window on which refitting is done
            refit data is cutoff (inclusive) to cutoff minus refit_window (exclusive)

    Parameters
    ----------
    refit_interval : difference of sktime time indices (int or timedelta), optional
        interval that needs to elapse after which the first update defaults to fit
        default = 0, i.e., always refits, never updates
    refit_window_size : difference of sktime time indices (int or timedelta), optional
        length of the data window to refit to in case update calls fit
        default = inf, i.e., refits to entire training data seen so far
    refit_window_lag : difference of sktime indices (int or timedelta), optional
        lag of the data window to refit to, w.r.t. cutoff, in case update calls fit
        default = 0, i.e., refit window ends with and includes cutoff
    """

    _delegate_name = "forecaster_"

    _tags = {"fit_is_empty": False}

    def __init__(
        self, forecaster, refit_interval=0, refit_window_size=None, refit_window_lag=0
    ):
        self.forecaster = forecaster
        self.forecaster_ = clone(forecaster)

        self.refit_interval = refit_interval
        self.refit_window_size = refit_window_size
        self.refit_window_lag = refit_window_lag

        super(UpdateRefitsEvery, self).__init__()

        self.clone_tags(forecaster)

        # fit must be executed to fit the wrapped estimator and remember the cutoff
        self.set_tags(fit_is_empty=False)

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
        self.last_fit_cutoff_ = self.cutoff
        estimator = self._get_delegate()
        estimator._fit(y=y, fh=fh, X=X)
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
        time_since_last_fit = self.cutoff - self.last_fit_cutoff_
        refit_window_size = self.refit_window_size
        refit_window_lag = self.refit_window_lag

        if time_since_last_fit >= self.refit_interval and update_params:
            if refit_window_size is not None or refit_window_lag != 0:
                y = get_window(y, window_length=refit_window_size, lag=refit_window_lag)
                X = get_window(X, window_length=refit_window_size, lag=refit_window_lag)
                fh = self._fh
            estimator._fit(y=y, X=X, fh=fh, update_params=update_params)
        else:
            estimator._update(y=y, X=X, update_params=update_params)
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
