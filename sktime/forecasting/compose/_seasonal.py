#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Seasonal reduction to multivariate."""

__author__ = ["fkiraly"]

__all__ = ["SeasonalReducer"]

import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.seasonality import _pivot_sp, _unpivot_sp


class SeasonalReducer(BaseForecaster):
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    estimator : sktime forecaster, descendant of BaseForecaster
        estimator to be used for the seasonal prediction
    sp : int
        seasonality used in reduction

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.compose import SeasonalReducer
    >>> from sktime.forecasting.trend import TrendForecaster
    >>>
    >>> y = load_airline()
    >>> f = SeasonalReducer(TrendForecaster(), sp=2)
    >>> f.fit(y, fh=[1, 2, 3])
    SeasonalReducer(...)
    >>> y_pred = f.predict()
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": False,
        "enforce_index_type": None,
        "handles-missing-data": True,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(self, forecaster, sp):

        self.forecaster = forecaster
        self.sp = sp

        super(SeasonalReducer, self).__init__()

        self.forecaster_ = forecaster.clone()

        tags_to_clone = {
            "ignores-exogeneous-X",
            "handles-missing-data"
            "capability:pred_int",
            "capability:insample",
            "capability:pred_int:insample",
        }
        self.clone_tags(forecaster, tags_to_clone)

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
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
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        f = self.forecaster_
        sp = self.sp

        y_pivot = _pivot_sp(y, sp=sp, anchor_side="end")

        if X is not None:
            X_pivot = _pivot_sp(X, sp=sp, anchor=y, anchor_side="end")
            X_pivot.columns = pd.RangeIndex(len(X_pivot.columns))
        else:
            X_pivot = None

        fh_ix = fh.to_absolute(self.cutoff).to_pandas()
        fh_df = pd.DataFrame(index=fh_ix, columns=y.columns)
        fh_df_pivot = _pivot_sp(fh_df, sp=sp, anchor=y, anchor_side="end")
        fh_pivot = ForecastingHorizon(fh_df_pivot.index, is_relative=False)

        if not f.get_tag("handles-missing-data"):
            y_pivot = y_pivot.fillna(method="bfill")
            if X is not None:
                X_pivot = X_pivot.fillna(method="bfill").fillna(method="ffill")

        f.fit(y=y_pivot, X=X_pivot, fh=fh_pivot)

        return self

    def _predict_method(self, fh, X=None, method="predict", **kwargs):
        """Template for predict-like methods, called from there."""
        f = self.forecaster_
        sp = self.sp
        _y = self._y

        if X is not None:
            X_pivot = _pivot_sp(X, sp=sp, anchor=_y)
        else:
            X_pivot = None

        y_pred_pivot = getattr(f, method)(X=X_pivot, **kwargs)

        y_pred = _unpivot_sp(y_pred_pivot, template=_y)
        expected_ix = fh.to_absolute(self.cutoff).to_pandas()

        y_pred = y_pred.reindex(expected_ix)
        return y_pred

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        return self._predict_method(fh=fh, X=X, method="predict")

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and possibly predict_interval

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        return self._predict_method(fh=fh, X=X, alpha=alpha, method="predict_quantiles")

    def _predict_interval(self, fh, X=None, coverage=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        return self._predict_method(
            fh=fh, X=X, coverage=coverage, method="predict_interval"
        )

    def _predict_var(self, fh, X=None, cov=False):
        """Forecast variance at future horizon.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on `cov` variable
            If cov=False:
                Column names are exactly those of `y` passed in `fit`/`update`.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are variance forecasts, for var in col index.
                A variance forecast for given variable and fh index is a predicted
                    variance for that variable and index, given observed data.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh, with additional levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
                Note: no covariance forecasts are returned between different variables.
        """
        return self._predict_method(fh=fh, X=X, cov=cov, method="predict_var")

    # todo 0.19.0 - remove legacy_interface
    def _predict_proba(self, fh, X, marginal=True, legacy_interface=None):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon (not optional)
            The forecasting horizon encoding the time stamps to forecast at.
            if has not been passed in fit, must be passed, not optional
        X : sktime time series object, optional (default=None)
                Exogeneous time series for the forecast
            Should be of same scitype (Series, Panel, or Hierarchical) as y in fit
            if self.get_tag("X-y-must-have-same-index"),
                X.index must contain fh.index and y.index both
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : tfp Distribution object
            if marginal=True:
                batch shape is 1D and same length as fh
                event shape is 1D, with length equal number of variables being forecast
                i-th (batch) distribution is forecast for i-th entry of fh
                j-th (event) index is j-th variable, order as y in `fit`/`update`
            if marginal=False:
                there is a single batch
                event shape is 2D, of shape (len(fh), no. variables)
                i-th (event dim 1) distribution is forecast for i-th entry of fh
                j-th (event dim 1) index is j-th variable, order as y in `fit`/`update`
        """
        kwargs = {"marginal": marginal, "legacy_interface": legacy_interface}
        return self._predict_method(fh=fh, X=X, method="predict_proba", **kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.compose._reduce import DirectReductionForecaster
        from sktime.forecasting.trend import TrendForecaster

        params1 = {"forecaster": TrendForecaster(), "sp": 3}
        params2 = {
            "forecaster": DirectReductionForecaster.create_test_instance(),
            "sp": 2,
        }

        return [params1, params2]
