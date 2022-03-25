# -*- coding: utf-8 -*-
"""Compositors that control stream and refitting behaviour of update."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

from sklearn import clone

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._delegate import _DelegatedForecaster


class UpdateRefitsEvery(BaseForecaster):
    """Refits periodically when update is called.

    If update is called with update_params=True and refit_interval or more has
        elapsed since the last fit, refits the forecaster instead.
        refit_window controls the lookback window on which refitting is done.

    Parameters
    ----------
    refit_interval : difference of sktime time indices, (int or timedelta), optional
        default = 0, i.e., always refits, never updates
    refit_window : difference of sktime time indices, (int or timedelta), optional
        default = inf, i.e., refits to entire training data seen so far
    """

    _delegate_name = "forecaster_"

    def __init__(self, forecaster, refit_interval=0, refit_window=None):
        self.forecaster = forecaster
        self.forecaster_ = clone(forecaster)

        self.refit_interval = refit_interval
        self.refit_window = refit_window

        super(UpdateRefitsEvery, self).__init__()

        self.clone_tags(forecaster)

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
        return estimator._fit(y=y, fh=fh, X=X)

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
        refit_window = self.refit_window

        # if refit_window is not None:
        #  subset

        if time_since_last_fit >= self.refit_interval and update_params:
            return estimator._fit(y=y, X=X, update_params=update_params)
        else:
            return estimator._update(y=y, X=X, update_params=update_params)
