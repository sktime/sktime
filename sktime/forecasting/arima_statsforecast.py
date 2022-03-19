# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements AutoARIMA model from StatsForecast"""

__author__ = ["FedericoGarza"]

from typing import Dict, Optional

import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("statsforecast", severity="warning")


class AutoARIMA(BaseForecaster):
    """StatsForecast AutoARIMA estimator.

    Returns best ARIMA model according to either AIC, AICc or BIC value.
    The function conducts a search over possible model within the order constraints provided.

    Parameters
    ----------
    d: int optional (default None)
        Order of first-differencing.
        If missing, will choose a value based on `test`.
    D: int optional (default None)
        Order of seasonal-differencing.
        If missing, will choose a value based on `season_test`.
    max_p: int (default 5)
        Maximum value of p.
    max_q: int (default 5)
        Maximum value of q.
    max_P: int (default 2)
        Maximum value of P.
    max_Q: int (default 2)
        Maximum value of Q.
    max_order: int (default 5)
        Maximum value of p+q+P+Q if model selection is not stepwise.
    max_d: int (default 2)
        Maximum number of non-seasonal differences
    max_D: int (default 1)
        Maximum number of seasonal differences
    start_p: int (default 2)
        Starting value of p in stepwise procedure.
    start_q: int (default 2)
        Starting value of q in stepwise procedure.
    start_P: int (default 1)
        Starting value of P in stepwise procedure.
    start_Q: int (default 1)
        Starting value of Q in stepwise procedure.
    stationary: bool (default False)
        If True, restricts search to stationary models.
    seasonal: bool (default True)
        If False, restricts search to non-seasonal models.
    ic: str (default 'aicc')
        Information criterion to be used in model selection.
    stepwise: bool (default True)
        If True, will do stepwise selection (faster).
        Otherwise, it searches over all models.
        Non-stepwise selection can be very slow,
        especially for seasonal models.
    nmodels: int (default 94)
        Maximum number of models considered in the stepwise search.
    trace: bool (default False)
        If True, the list of ARIMA models considered will be reported.
    approximation: bool optional (default None)
        If True, estimation is via conditional sums of squares
        and the information criteria used for model
        selection are approximated.
        The final model is still computed using
        maximum likelihood estimation.
        Approximation should be used for long time series
        or a high seasonal period to avoid excessive computation times.
    method: str optional (default None)
        fitting method: maximum likelihood or minimize conditional
        sum-of-squares.
        The default (unless there are missing values)
        is to use conditional-sum-of-squares to find starting values,
        then maximum likelihood. Can be abbreviated.
    truncate: bool optional (default None)
        An integer value indicating how many observations
        to use in model selection.
        The last truncate values of the series are
        used to select a model when truncate is not None
        and approximation=True.
        All observations are used if either truncate=None
        or approximation=False.
    test: str (default 'kpss')
        Type of unit root test to use. See ndiffs for details.
    test_kwargs: str optional (default None)
        Additional arguments to be passed to the unit root test.
    seasonal_test: str (default 'seas')
        This determines which method is used to select the number
        of seasonal differences.
        The default method is to use a measure of seasonal
        strength computed from an STL decomposition.
        Other possibilities involve seasonal unit root tests.
    seasonal_test_kwargs: dict optional (default None)
        Additional arguments to be passed to the seasonal
        unit root test. See nsdiffs for details.
    allowdrift: bool (default True)
        If True, models with drift terms are considered.
    allowmean: bool (default True)
        If True, models with a non-zero mean are considered.
    blambda: float optional (default None)
        Box-Cox transformation parameter.
        If lambda="auto", then a transformation is automatically
        selected using BoxCox.lambda.
        The transformation is ignored if None.
        Otherwise, data transformed before model is estimated.
    biasadj: bool (default False)
        Use adjusted back-transformed mean for Box-Cox transformations.
        If transformed data is used to produce forecasts and fitted values,
        a regular back transformation will result in median forecasts.
        If biasadj is True, an adjustment will be made to produce
        mean forecasts and fitted values.
    parallel: bool (default False)
        If True and stepwise = False, then the specification search
        is done in parallel.
        This can give a significant speedup on multicore machines.
    num_cores: int (default 2)
        Allows the user to specify the amount of parallel processes to be used
        if parallel = True and stepwise = False.
        If None, then the number of logical cores is
        automatically detected and all available cores are used.
    period: int (default 1)
        Number of observations per unit of time.
        For example 24 for Hourly data.

    Notes
    -----
    * This implementation is a mirror of Hyndman's forecast::auto.arima.
    * This implementation is a wrapper of AutoARIMA from StatsForecast.

    References
    ----------
    [1] hhttps://github.com/robjhyndman/forecast
    """
    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": True,  # does forecaster implement predict_quantiles?
        # deprecated and will be renamed to capability:predict_quantiles in 0.11.0
    }

    def __init__(
        self,
        d: Optional[int] = None,
        D: Optional[int] = None,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        max_order: int = 5,
        max_d: int = 2,
        max_D: int = 1,
        start_p: int = 2,
        start_q: int = 2,
        start_P: int = 1,
        start_Q: int = 1,
        stationary: bool = False,
        seasonal: bool = True,
        ic: str = 'aicc',
        stepwise: bool = True,
        nmodels: int = 94,
        trace: bool = False,
        approximation: Optional[bool] = None,
        method: Optional[str] = None,
        truncate: Optional[bool] = None,
        test: str = 'kpss',
        test_kwargs: Optional[str] = None,
        seasonal_test: str = 'seas',
        seasonal_test_kwargs: Optional[Dict] = None,
        allowdrift: bool = True,
        allowmean: bool = True,
        blambda: Optional[float] = None,
        biasadj: bool = False,
        parallel: bool = False,
        num_cores: int = 2,
        period: int = 1
    ):
        _check_soft_dependencies("statsforecast", severity="error", object=self)

        self.d=d
        self.D=D
        self.max_p=max_p
        self.max_q=max_q
        self.max_P=max_P
        self.max_Q=max_Q
        self.max_order=max_order
        self.max_d=max_d
        self.max_D=max_D
        self.start_p=start_p
        self.start_q=start_q
        self.start_P=start_P
        self.start_Q=start_Q
        self.stationary=stationary
        self.seasonal=seasonal
        self.ic=ic
        self.stepwise=stepwise
        self.nmodels=nmodels
        self.trace=trace
        self.approximation=approximation
        self.method=method
        self.truncate=truncate
        self.test=test
        self.test_kwargs=test_kwargs
        self.seasonal_test=seasonal_test
        self.seasonal_test_kwargs=seasonal_test_kwargs
        self.allowdrift=allowdrift
        self.allowmean=allowmean
        self.blambda=blambda
        self.biasadj=biasadj
        self.parallel=parallel
        self.num_cores=num_cores
        self.period=period

        super(AutoARIMA, self).__init__()

    def _instantiate_model(self):
        # import inside method to avoid hard dependency
        from statsforecast.arima import AutoARIMA as _AutoARIMA

        return _AutoARIMA(
                    d=self.d,
                    D=self.D,
                    max_p=self.max_p,
                    max_q=self.max_q,
                    max_P=self.max_P,
                    max_Q=self.max_Q,
                    max_order=self.max_order,
                    max_d=self.max_d,
                    max_D=self.max_D,
                    start_p=self.start_p,
                    start_q=self.start_q,
                    start_P=self.start_P,
                    start_Q=self.start_Q,
                    stationary=self.stationary,
                    seasonal=self.seasonal,
                    ic=self.ic,
                    stepwise=self.stepwise,
                    nmodels=self.nmodels,
                    trace=self.trace,
                    approximation=self.approximation,
                    method=self.method,
                    truncate=self.truncate,
                    test=self.test,
                    test_kwargs=self.test_kwargs,
                    seasonal_test=self.seasonal_test,
                    seasonal_test_kwargs=self.seasonal_test_kwargs,
                    allowdrift=self.allowdrift,
                    allowmean=self.allowmean,
                    blambda=self.blambda,
                    biasadj=self.biasadj,
                    parallel=self.parallel,
                    num_cores=self.num_cores,
                    period=self.period
                )

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
        self._forecaster = self._instantiate_model()
        self._forecaster.fit(y.values, X.values if X is not None else X)

        return self

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
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        # distinguish between in-sample and out-of-sample prediction
        fh_oos = fh.to_out_of_sample(self.cutoff)
        fh_ins = fh.to_in_sample(self.cutoff)

        # all values are out-of-sample
        if fh.is_all_out_of_sample(self.cutoff):
            return self._predict_fixed_cutoff(fh_oos, X=X)

        # all values are in-sample
        elif fh.is_all_in_sample(self.cutoff):
            return self._predict_in_sample(fh_ins, X=X)

        # both in-sample and out-of-sample values
        else:
            y_ins = self._predict_in_sample(fh_ins, X=X)
            y_oos = self._predict_fixed_cutoff(fh_oos, X=X)
            return y_ins.append(y_oos)

    def _predict_in_sample(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Generate in sample predictions.
        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).
        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        """
        #initialize return objects
        fh_abs = fh.to_absolute(self.cutoff).to_numpy()
        fh_idx = fh.to_indexer(self.cutoff, from_cutoff=False)
        y_pred = pd.Series(index=fh_abs)

        result = self._forecaster.predict_in_sample()
        y_pred.loc[fh_abs] = result['mean'].values[fh_idx]

        if return_pred_int:
            pred_ints = []
            for a in alpha:
                pred_int = pd.DataFrame(index=fh_abs, columns=["lower", "upper"])
                result = self._forecaster.predict_in_sample(level=int(100 * a))
                pred_int.loc[fh_abs] = result.drop('mean', axis=1).values[fh_idx, :]
                pred_ints.append(pred_int)
            return y_pred, pred_ints
        
        return y_pred

    def _predict_fixed_cutoff(
        self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA
    ):
        """Make predictions out of sample.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).

        Returns
        -------
        y_pred : pandas.Series
        Returns series of predicted values.
        """
        n_periods = int(fh.to_relative(self.cutoff)[-1])
        result = self._forecaster.predict(
            h=n_periods,
            X=X.values if X is not None else X,
        )

        fh_abs = fh.to_absolute(self.cutoff)
        fh_idx = fh.to_indexer(self.cutoff)
        mean = pd.Series(result['mean'].values[fh_idx], index=fh_abs)
        if return_pred_int:
            pred_ints = []
            for a in alpha:
                result = self._forecaster.predict(
                    h=n_periods,
                    X=X.values if X is not None else X,
                    level=int(100 * a),
                )
                pred_int = result.drop('mean', axis=1).values
                pred_int = pd.DataFrame(
                    pred_int[fh_idx, :], index=fh_abs, columns=["lower", "upper"]
                )
                pred_ints.append(pred_int)
            return mean, pred_ints
        else:
            return pd.Series(mean, index=fh_abs)

    def _predict_interval(self, fh, X=None, coverage=0.90):
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
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
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
        # initializaing cutoff and fh related info
        cutoff = self.cutoff
        fh_oos = fh.to_out_of_sample(cutoff)
        fh_ins = fh.to_in_sample(cutoff)
        fh_is_in_sample = fh.is_all_in_sample(cutoff)
        fh_is_oosample = fh.is_all_out_of_sample(cutoff)

        # prepare the return DataFrame - empty with correct cols
        var_names = ["Coverage"]
        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(columns=int_idx)

        kwargs = {"X": X, "return_pred_int": True, "alpha": coverage}
        # all values are out-of-sample
        if fh_is_oosample:
            _, y_pred_int = self._predict_fixed_cutoff(fh_oos, **kwargs)

        # all values are in-sample
        elif fh_is_in_sample:
            _, y_pred_int = self._predict_in_sample(fh_ins, **kwargs)

        # if all in-sample/out-of-sample, we put y_pred_int in the required format
        if fh_is_in_sample or fh_is_oosample:
            # needs to be replaced, also seems duplicative, identical to part A
            for intervals, a in zip(y_pred_int, coverage):
                pred_int[("Coverage", a, "lower")] = intervals["lower"]
                pred_int[("Coverage", a, "upper")] = intervals["upper"]
            return pred_int

        # both in-sample and out-of-sample values (we reach this line only then)
        # in this case, we additionally need to concat in and out-of-sample returns
        _, y_ins_pred_int = self._predict_in_sample(fh_ins, **kwargs)
        _, y_oos_pred_int = self._predict_fixed_cutoff(fh_oos, **kwargs)
        for ins_int, oos_int, a in zip(y_ins_pred_int, y_oos_pred_int, coverage):
            pred_int[("Coverage", a, "lower")] = ins_int.append(oos_int)["lower"]
            pred_int[("Coverage", a, "upper")] = ins_int.append(oos_int)["upper"]

        return pred_int

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
        params = {"approximation": True, "max_p": 4, "max_Q": 1}
        return params
