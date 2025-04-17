# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements compositor for different forecast by fh index."""

from sktime.forecasting.base import BaseForecaster

__author__ = ["fkiraly"]
__all__ = ["FhPlexForecaster"]

import pandas as pd

from sktime.datatypes._utilities import get_slice

PANDAS_TS_MTYPES = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]


class FhPlexForecaster(BaseForecaster):
    """Uses different parameters by forecasting horizon element.

    When provided with forecasting horizon [f1, f2, ..., fn],
    will fit forecaster with fh=f1 and parameters fh_params[f1] to forecast f1,
    forecaster with fh=f2 and parameters fh_params[f2] to forecast f2, etc.

    To use different estimators per horizon, combine ``FhPlexForecaster`` with
    one of ``MultiplexForecaster`` and ``MultiplexTransformer``.

    Parameters
    ----------
    forecaster : sktime compatible forecaster
    fh_params : dict, list, callable, or str that eval-defines a callable
        specifies forecaster to use per fh element
        dict: keys = fh elements, values = param dict for forecaster
        list: i-th entry is forecaster param dict for i-th fh element
        callable: maps fh element to forecaster param dict
        str: eval(fh_params) must define a lambda that maps fh element to param dict
        param dict need not be complete, only overrides for ``forecaster`` params
    fh_lookup : str, one of "relative" (default), "absolute", or "as-is"
        specifies fh elements used in dict or callable
        if "relative", fh will be coerced to relative ForecastingHorizon
        if "absolute", fh will be coerced to absolute ForecastingHorizon
        if "as-is", fh will be coerced to ForecastingHorizon (but not relative/absolute)
    fh_contiguous : bool, default=False
        whether fh in inner loops are enforced to be contiguous
        False: forecaster with fh_params[fN] is asked to forecast fN and only fN
        True: forecaster with fh_params[fN] is asked to forecast 1, 2, ..., fN
            and the output is then subset to the forecast of fN
            this is required if the forecaster can only forecast contiguous horizons
        CAUTION: if using grid search inside, then ``True`` will cause the
        tuning metric to be evaluated on horizons 1, 2, ..., fN, not just fN

    Attributes
    ----------
    forecasters_ : dict of sktime forecaster
        keys are fh elements (coerced according to ``fh_lookup``)
        entries are clones of ``forecaster`` used for fitting and forecasting

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import FhPlexForecaster

    Simple example - same parameters per fh element
    >>> y = load_airline()
    >>> f = FhPlexForecaster(NaiveForecaster())
    >>> f.fit(y, fh=[1, 2, 3])
    FhPlexForecaster(...)
    >>> # get individual fitted forecasters
    >>> f.forecasters_  # doctest: +SKIP
    {1: NaiveForecaster(), 2: NaiveForecaster(), 3: NaiveForecaster()}
    >>> fitted_params = f.get_fitted_params()  # or via get_fitted_params
    >>> y_pred = f.predict()

    Simple example - different parameters per fh element
    >>> y = load_airline()
    >>> fh_params = [{}, {"strategy": "last"}, {"strategy": "mean"}]
    >>> f = FhPlexForecaster(NaiveForecaster(), fh_params=fh_params)
    >>> f.fit(y, fh=[1, 2, 3])
    FhPlexForecaster(...)
    >>> # get individual fitted forecasters
    >>> f.forecasters_  # doctest: +SKIP
    {1: NaiveForecaster(), 2: NaiveForecaster(), 3: NaiveForecaster(strategy='mean')}
    >>> y_pred = f.predict()
    """

    _tags = {
        "authors": "fkiraly",
        "requires-fh-in-fit": True,
        "capability:missing_values": True,
        "scitype:y": "both",
        "y_inner_mtype": PANDAS_TS_MTYPES,
        "X_inner_mtype": PANDAS_TS_MTYPES,
        "fit_is_empty": False,
        "capability:pred_int": True,
    }

    def __init__(
        self, forecaster, fh_params=None, fh_lookup="relative", fh_contiguous=False
    ):
        self.forecaster = forecaster
        self.fh_params = fh_params
        self.fh_lookup = fh_lookup
        self.fh_contiguous = fh_contiguous

        super().__init__()

    @property
    def _forecasters(self):
        """Forecasters turned into name/est tuples."""
        fh_keys = self._get_fh_keys(self._fh)
        f_tupl = [(str(k), self.forecasters_[k]) for k in fh_keys]
        return f_tupl

    @property
    def _plexfun(self):
        """Get function that returns parameter dict given fh key index."""
        fh_params = self.fh_params

        if fh_params is None:

            def ret_param(ix):
                return {}

        elif isinstance(fh_params, (list, dict)):

            def ret_param(ix):
                return fh_params[ix]

        elif isinstance(fh_params, str):
            ret_param = eval(fh_params)
        else:
            ret_param = fh_params

        return ret_param

    def _get_fh_keys(self, fh):
        """Get keys used for self.forecasters_, from fh, given fh_lookup."""
        fh_lookup = self.fh_lookup

        if fh_lookup == "relative":
            fh = fh.to_relative(self.cutoff)
        elif fh_lookup == "absolute":
            fh = fh.to_absolute(self.cutoff)

        return fh

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
        fh_params = self.fh_params
        fh_contiguous = self.fh_contiguous

        if fh_contiguous:
            fh_rel = fh.to_relative(self.cutoff)

        fh_keys = self._get_fh_keys(fh)

        self.forecasters_ = {}

        for i, fh_key in enumerate(fh_keys):
            if isinstance(fh_params, list):
                ix = i
            else:
                ix = fh_key

            if not fh_contiguous or fh_rel[i] <= 0:
                fh_i = [fh_key]
            else:
                fh_rel = fh.to_relative(self.cutoff)
                fh_i = range(1, fh_rel[i] + 1)

            params_ix = self._plexfun(ix)
            f_ix = self.forecaster.clone().set_params(**params_ix)
            self.forecasters_[fh_key] = f_ix.fit(y=y, X=X, fh=fh_i)

        return self

    def _get_preds(self, fh_keys, method="predict", **kwargs):
        """Get prediction DataFrame for method."""
        fh_contiguous = self.fh_contiguous

        if fh_contiguous:
            fh_abs = fh_keys.to_absolute(self.cutoff)

        fh_keys = self._get_fh_keys(fh_keys)

        y_preds = []

        for i, fh_key in enumerate(fh_keys):
            fh_method = getattr(self.forecasters_[fh_key], method)
            if not fh_contiguous:
                y_preds += [fh_method(**kwargs)]
            else:
                y_pred = fh_method(**kwargs)
                y_pred_slice = get_slice(y_pred, fh_abs[i])
                y_preds += [y_pred_slice]

        y_pred = pd.concat(y_preds, axis=0)
        return y_pred.sort_index()

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
        y_pred : pd.DataFrame
            Point predictions
        """
        y_pred = self._get_preds(fh, "predict", X=X)
        return y_pred

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
        fh_keys = self._get_fh_keys(self._fh)

        for fh_key in fh_keys:
            fcst = self.forecasters_[fh_key]
            fcst.update(y=y, X=X, update_params=update_params)

        return self

    def _update_predict_single(self, y, fh, X=None, update_params=True):
        """Update forecaster and then make forecasts.

        Implements default behaviour of calling update and predict sequentially, but can
        be overwritten by subclasses to implement more efficient updating algorithms
        when available.
        """
        y_pred = self._get_preds(
            fh, "update_predict_single", y=y, X=X, update_params=update_params
        )
        return y_pred

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
        y_pred = self._get_preds(fh, "predict_quantiles", X=X, alpha=alpha)
        return y_pred

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
                    in the same order as in input ``coverage``.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        y_pred = self._get_preds(fh, "predict_interval", X=X, coverage=coverage)
        return y_pred

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
        pred_var : pd.DataFrame, format dependent on ``cov`` variable
            If cov=False:
                Column names are exactly those of ``y`` passed in ``fit``/``update``.
                    For nameless formats, column index will be a RangeIndex.
                Row index is fh. Entries are variance forecasts, for var in col index.
            If cov=True:
                Column index is a multiindex: 1st level is variable names (as above)
                    2nd level is fh.
                Row index is fh.
                Entries are (co-)variance forecasts, for var in col index, and
                    covariance between time index in row and col.
        """
        y_pred = self._get_preds(fh, "predict_var", X=X, cov=cov)
        return y_pred

        # todo - implement concat
        # def _predict_proba(self, fh, X, marginal=True):
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
        fh_keys = self.forecasters_.keys()

        fitted_params = {}
        for fh_key in fh_keys:
            fh_key_params = self.forecasters_[fh_key].get_fitted_params(deep=True)
            for k in fh_key_params:
                fitted_params[f"{fh_key}__{k}"] = fh_key_params[k]

        return fitted_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster

        # mean and drift are inefficient, for now use last only
        # naive_m = ["last", "mean", "drift"]

        naive_m = ["last", "last", "last"]
        naive_list = [{"strategy": x} for x in naive_m * 20]
        naive_dict = {k: naive_list[k % 3] for k in range(-50, 10)}

        naive_str = "lambda ix: {'strategy': ['last', 'last', 'last'][ix % 3]}"

        f = NaiveForecaster()

        params1 = {"forecaster": f, "fh_params": naive_list}
        params2 = {
            "forecaster": f,
            "fh_params": naive_dict,
            "fh_lookup": "relative",
            "fh_contiguous": True,
        }
        params3 = {"forecaster": f, "fh_params": naive_str, "fh_lookup": "relative"}

        return [params1, params2, params3]
