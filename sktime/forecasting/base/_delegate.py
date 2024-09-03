"""Delegator mixin that delegates all methods to wrapped forecaster.

Useful for building estimators where all but one or a few methods are delegated. For
that purpose, inherit from this estimator and then override only the methods that
are not delegated.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["_DelegatedForecaster"]

from sktime.forecasting.base import BaseForecaster


class _DelegatedForecaster(BaseForecaster):
    """Delegator mixin that delegates all methods to wrapped forecaster.

    Delegates inner forecaster methods to a wrapped estimator.
        Wrapped estimator is value of attribute with name self._delegate_name.
        By default, this is "estimator_", i.e., delegates to self.estimator_
        To override delegation, override _delegate_name attribute in child class.

    Delegates the following inner underscore methods:
        _fit, _predict, _update
        _predict_interval, _predict_quantiles, _predict_var, _predict_proba

    Does NOT delegate get_params, set_params.
        get_params, set_params will hence use one additional nesting level by default.

    Does NOT delegate or copy tags, this should be done in a child class if required.
    """

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "estimator_"

    def _get_delegate(self):
        return getattr(self, self._delegate_name)

    def _set_delegated_tags(self, delegate=None):
        """Set delegated tags, only tags for boilerplate control.

        Writes tags to self.
        Can be used by descendant classes to set dependent tags.
        Makes safe baseline assumptions about tags, which can be overwritten.

        * data mtype tags are set to the most general value.
          This is to ensure that conversion is left to the inner estimator.
        * packaging tags such as "author" or "python_dependencies" are not cloned.
        * other boilerplate tags are cloned.

        Parameters
        ----------
        delegate : object, optional (default=None)
            object to get tags from, if None, uses self._get_delegate()

        Returns
        -------
        self : reference to self
        """
        from sktime.datatypes import ALL_TIME_SERIES_MTYPES

        if delegate is None:
            delegate = self._get_delegate()

        TAGS_TO_DELEGATE = [
            "requires-fh-in-fit",
            "handles-missing-data",
            "ignores-exogeneous-X",
            "capability:insample",
            "capability:pred_int",
            "capability:pred_int:insample",
            "handles-missing-data",
            "requires-fh-in-fit",
            "X-y-must-have-same-index",
            "enforce_index_type",
            "fit_is_empty",
        ]

        TAGS_TO_SET = {
            "scitype:y": "both",
            "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
            "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        }

        self.clone_tags(delegate, tag_names=TAGS_TO_DELEGATE)
        self.set_tags(**TAGS_TO_SET)

        return self

    def _fit(self, y, X, fh):
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
        estimator = self._get_delegate()
        estimator.fit(y=y, fh=fh, X=X)
        return self

    def _predict(self, fh, X):
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
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        estimator = self._get_delegate()
        return estimator.predict(fh=fh, X=X)

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
        estimator.update(y=y, X=X, update_params=update_params)
        return self

    def _update_predict_single(self, y, fh, X=None, update_params=True):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict sequentially, but can
        be overwritten by subclasses to implement more efficient updating algorithms
        when available.
        """
        estimator = self._get_delegate()
        return estimator.update_predict_single(
            y=y, fh=fh, X=X, update_params=update_params
        )

    def _predict_quantiles(self, fh, X, alpha):
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
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float (guaranteed not None and floats in [0,1] interval)
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        pred_quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the quantile forecasts for each alpha.
                Quantile forecasts are calculated for each a in alpha.
            Row index is fh. Entries are quantile forecasts, for var in col index,
                at quantile probability in second-level col index, for each row index.
        """
        estimator = self._get_delegate()
        return estimator.predict_quantiles(fh=fh, X=X, alpha=alpha)

    def _predict_interval(self, fh, X, coverage):
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
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        estimator = self._get_delegate()
        return estimator.predict_interval(fh=fh, X=X, coverage=coverage)

    def _predict_var(self, fh, X=None, cov=False):
        """Forecast variance at future horizon.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        cov : bool, optional (default=False)
            if True, computes covariance matrix forecast.
            if False, computes marginal variance forecasts.

        Returns
        -------
        pred_var : pd.DataFrame, format dependent on `cov` variable
            If cov=False:
                Column names are exactly those of `y` passed in `fit`/`update`.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh. Entries are variance forecasts, for var in col index.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
        """
        estimator = self._get_delegate()
        return estimator.predict_var(fh=fh, X=X, cov=cov)

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        private _predict_proba containing the core logic, called from predict_proba

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
            if marginal=True, will be marginal distribution by time point
            if marginal=False and implemented by method, will be joint
        """
        estimator = self._get_delegate()
        return estimator.predict_proba(fh=fh, X=X, marginal=marginal)

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        estimator = self._get_delegate()
        return estimator.get_fitted_params()
