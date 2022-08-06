# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements ARDL Model as interface to statsmodels."""


_all_ = ["ARDL"]
__author__ = ["kcc-lion"]

"""
Extension template for forecasters.

Purpose of this implementation template:
    quick implementation of new estimators following the template
    NOT a concrete class to import! This is NOT a base class or concrete class!
    This is to be used as a "fill-in" coding template.

How to use this implementation template to implement a new estimator:
- make a copy of the template in a suitable location, give it a descriptive name.
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y, cutoff, _fh,
    _cutoff, _converter_store_y, forecasters_, _tags, _tags_dynamic, _is_vectorized
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details: https://www.sktime.org/en/stable/developer_guide/add_estimators.html

Mandatory implements:
    fitting         - _fit(self, y, X=None, fh=None)
    forecasting     - _predict(self, fh=None, X=None)

Optional implements:
    updating                    - _update(self, y, X=None, update_params=True):
    predicting quantiles        - _predict_quantiles(self, fh, X=None, alpha=None)
    OR predicting intervals     - _predict_interval(self, fh, X=None, coverage=None)
    predicting variance         - _predict_var(self, fh, X=None, cov=False)
    distribution forecast       - _predict_proba(self, fh, X=None)
    fitted parameter inspection - get_fitted_params()

Testing - implement if sktime forecaster (not needed locally):
    get default parameters for test instance(s) - get_test_params()
"""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to sktime should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright


# todo: add any necessary imports here
import pandas as pd
from statsmodels.tsa.ardl import ARDL as _ARDL
from statsmodels.tsa.ardl import ardl_select_order as _ardl_select_order

from sktime.forecasting.base.adapters import _StatsModelsAdapter
from sktime.forecasting.base.adapters._statsmodels import _coerce_int_to_range_index

class ARDL(_StatsModelsAdapter):
    """Autoregressive Distributed Lag (ARDL) Model. todo: write docstring.

    Direct interface for statsmodels.tsa.ardl.ARDL

    Parameters todo: overwrite parameters
    ----------
    lags : {int, list[int]}, optional
        Only considered if auto_ardl is False
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
    order : {int, sequence[int], dict}
        Only considered of auto_ardl is False
        If int, uses lags 0, 1, ..., order  for all exog variables. If
        sequence[int], uses the ``order`` for all variables. If a dict,
        applies the lags series by series. If ``exog`` is anything other
        than a DataFrame, the keys are the column index of exog (e.g., 0,
        1, ...). If a DataFrame, keys are column names.
    fixed : array_like
        Additional fixed regressors that are not lagged.
    causal : bool, optional
        Whether to include lag 0 of exog variables.  If True, only includes
        lags 1, 2, ...
    trend : {'n', 'c', 't', 'ct'}, optional
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

        The default is 'c'.

    seasonal : bool, optional
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    deterministic : DeterministicProcess, optional
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.
    hold_back : {None, int}, optional
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}, optional
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : {"none", "drop", "raise"}, optional
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.
    cov_type : str
        The covariance estimator to use. The most common choices are listed
        below.  Supports all covariance estimators that are available
        in ``OLS.fit``.

        * 'nonrobust' - The class OLS covariance estimator that assumes
          homoskedasticity.
        * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
          (or Eiker-Huber-White) covariance estimator. `HC0` is the
          standard implementation.  The other make corrections to improve
          the finite sample performance of the heteroskedasticity robust
          covariance estimator.
        * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
          estimation. Supports cov_kwds.

          - `maxlags` integer (required) : number of lags to use.
          - `kernel` callable or str (optional) : kernel
              currently available kernels are ['bartlett', 'uniform'],
              default is Bartlett.
          - `use_correction` bool (optional) : If true, use small sample
              correction.
    cov_kwds : dict, optional
        A dictionary of keyword arguments to pass to the covariance
        estimator. `nonrobust` and `HC#` do not support cov_kwds.
    use_t : bool, optional
        A flag indicating that inference should use the Student's t
        distribution that accounts for model degree of freedom.  If False,
        uses the normal distribution. If None, defers the choice to
        the cov_type. It also removes degree of freedom corrections from
        the covariance estimator when cov_type is 'nonrobust'.
    auto_ardl : bool, optional
        A flag indicating whether the number of lags should be determined automatically
    maxlag : int
        Only considered if auto_ardl is True.
        The maximum lag to consider for the endogenous variable.
    maxorder : {int, dict}
        Only considered if auto_ardl is True.
        If int, sets a common max lag length for all exog variables. If
        a dict, then sets individual lag length. They keys are column names
        if exog is a DataFrame or column indices otherwise.
    ic : {"aic", "bic", "hqic"}
        Only considered if auto_ardl is True.
        The information criterion to use in model selection.
    glob : bool
        Only considered if auto_ardl is True.
        Whether to consider all possible submodels of the largest model
        or only if smaller order lags must be included if larger order
        lags are.  If ``True``, the number of model considered is of the
        order 2**(maxlag + k * maxorder) assuming maxorder is an int. This
        can be very large unless k and maxorder are bot relatively small.
        If False, the number of model considered is of the order
        maxlag*maxorder**k which may also be substantial when k and maxorder
        are large.
    X_oos : array_like
        An array containing out-of-sample values of the exogenous
        variables. Must have the same number of columns as the X
        and at least as many rows as the number of out-of-sample forecasts.
    fixed_oos : array_like
        An array containing out-of-sample values of the fixed variables.
        Must have the same number of columns as the fixed array
        and at least as many rows as the number of out-of-sample forecasts.
    dynamic : {bool, int, str, datetime, Timestamp}, optional
        Integer offset relative to `start` at which to begin dynamic
        prediction. Prior to this observation, true endogenous values
        will be used for prediction; starting with this observation and
        continuing through the end of prediction, forecasted endogenous
        values will be used instead. Datetime-like objects are not
        interpreted as offsets. They are instead used to find the index
        location of `dynamic` which is then used to to compute the offset.

    Notes
    -----
    The full specification of an ARDL is

    .. math ::

       Y_t = \delta_0 + \delta_1 t + \delta_2 t^2
             + \sum_{i=1}^{s-1} \gamma_i I_{[(\mod(t,s) + 1) = i]}
             + \sum_{j=1}^p \phi_j Y_{t-j}
             + \sum_{l=1}^k \sum_{m=0}^{o_l} \beta_{l,m} X_{l, t-m}
             + Z_t \lambda
             + \epsilon_t

    where :math:`\delta_\bullet` capture trends, :math:`\gamma_\bullet`
    capture seasonal shifts, s is the period of the seasonality, p is the
    lag length of the endogenous variable, k is the number of exogenous
    variables :math:`X_{l}`, :math:`o_l` is included the lag length of
    :math:`X_{l}`, :math:`Z_t` are ``r`` included fixed regressors and
    :math:`\epsilon_t` is a white noise shock. If ``causal`` is ``True``,
    then the 0-th lag of the exogenous variables is not included and the
    sum starts at ``m=1``.

    See Also
    --------
    statsmodels.tsa.ar_model.AutoReg
        Autoregressive model estimation with optional exogenous regressors
    statsmodels.tsa.ardl.UECM
        Unconstrained Error Correction Model estimation
    statsmodels.tsa.statespace.sarimax.SARIMAX
        Seasonal ARIMA model estimation with optional exogenous regressors
    statsmodels.tsa.arima.model.ARIMA
        ARIMA model estimation

    Examples
    --------
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # todo: define the forecaster scitype by setting the tags
    #  the "forecaster scitype" is determined by the tags
    #   scitype:y - the expected input scitype of y - univariate or multivariate or both
    #  when changing scitype:y to multivariate or both:
    #   y_inner_mtype should be changed to pd.DataFrame
    # other tags are "safe defaults" which can usually be left as-is
    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
        "python_version": None,  # PEP 440 python version specifier to limit versions
    }
    #  in case of inheritance, concrete class should typically set tags
    #  alternatively, descendants can set tags in __init__ (avoid this if possible)

    # todo: add any hyper-parameters and components to constructor
    def __init__(
            self,
            lags=None,
            order=0,
            fixed=None,
            causal=False,
            trend='c',
            seasonal=False,
            deterministic=None,
            hold_back=None,
            period=None,
            missing='none',
            cov_type='nonrobust',
            cov_kwds=None,
            use_t=True,
            auto_ardl=False,
            maxlag=None,
            maxorder=None,
            ic='bic',
            glob=False,
            fixed_oos=None,
            X_oos=None,
            dynamic=False
    ):

        # Model Params
        self.lags = lags
        self.order = order
        self.fixed = fixed
        self.causal = causal
        self.trend = trend
        self.seasonal = seasonal
        self.deterministic = deterministic
        self.hold_back = hold_back
        self.period = period
        self.missing = missing

        # Fit Params
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.use_t = use_t

        # Predict Params
        self.fixed_oos = fixed_oos
        self.X_oos = X_oos
        self.dynamic = dynamic

        # Auto ARDL params
        self.auto_ardl = auto_ardl
        self.maxlag = maxlag
        self.ic = ic
        self.glob = glob
        self.maxorder = maxorder

        if not self.auto_ardl:
            assert self.lags is not None

        if self.auto_ardl and self.lags is not None:
            raise ValueError('lags should not be specified if aut_ardl is True')

        super(ARDL, self).__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement this, mandatory
    def _fit(self, y, X, fh=None):
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
            A 1-d endogenous response variable. The dependent variable.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.
            Exogenous variables to include in the model. Either a DataFrame or
            an 2-d array-like structure that can be converted to a NumPy array.

        Returns
        -------
        self : reference to self
        """
        # statsmodels does not support the pd.Int64Index as required,
        # so we coerce them here to pd.RangeIndex
        if isinstance(y, pd.Series) and y.index.is_integer():
            y, X = _coerce_int_to_range_index(y, X)

        if not self.auto_ardl:
            self._forecaster = _ARDL(
                endog=y,
                lags=self.lags,
                exog=X,
                order=self.order,
                trend=self.trend,
                fixed=self.fixed,
                causal=self.causal,
                seasonal=self.seasonal,
                deterministic=self.deterministic,
                hold_back=self.hold_back,
                period=self.period,
                missing=self.missing
            )

            self._fitted_forecaster = self._forecaster.fit(
                cov_type=self.cov_type,
                cov_kwds=self.cov_kwds,
                use_t=self.use_t
            )
        else:
            self._forecaster = _ardl_select_order(
                endog=y,
                maxlag=self.maxlag,
                exog=X,
                maxorder=self.maxorder,
                trend=self.trend,
                fixed=self.fixed,
                causal=self.causal,
                ic=self.ic,
                glob=self.glob,
                seasonal=self.seasonal,
                deterministic=self.deterministic,
                hold_back=self.hold_back,
                period=self.period,
                missing=self.missing
            )

            self._fitted_forecaster = self._forecaster.model.fit(
                cov_type=self.cov_type,
                cov_kwds=self.cov_kwds,
                use_t=self.use_t
            )
        # implement here
        # IMPORTANT: avoid side effects to y, X, fh
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #  if used, estimators should be cloned to attributes ending in "_"
        #  the clones, not the originals shoudld be used or fitted if needed
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (y, X) or forecasting-horizon-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit
        return self

    def summary(self):
        """Get a summary of the fitted forecaster."""
        self.check_is_fitted()
        return self._fitted_forecaster.summary()

    def hessian(self):
        """Get the hessian of the fitted forecaster."""
        self.check_is_fitted()
        return self._forecaster.hessian(self._fitted_forecaster.params)

    def information(self):
        """Get the fisher information matrix of the fitted forecaster."""
        self.check_is_fitted()
        return self._forecaster.information(self._fitted_forecaster.params)

    def loglike(self):
        """Get the loglikelihood of the fitted forecaster."""
        self.check_is_fitted()
        return self._forecaster.loglike(self._fitted_forecaster.params)

    def score(self):
        """Get the loglikelihood of the fitted forecaster."""
        self.check_is_fitted()
        return self._forecaster.score(self._fitted_forecaster.params)


    # todo: implement this, mandatory
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
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        # statsmodels requires zero-based indexing starting at the
        # beginning of the training series when passing integers
        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]
        if X is not None:
            X_oos = X[~X.index.isin(self._X.index)]
        else:
            X_oos = None
        # statsmodels forecasts all periods from start to end of forecasting
        # horizon, but only return given time points in forecasting horizon
        valid_indices = fh.to_absolute(self.cutoff).to_pandas()

        y_pred = self._fitted_forecaster.predict(start=start, end=end, exog=self._X, exog_oos=X_oos, fixed_oos=self.fixed_oos)
        return y_pred.loc[valid_indices]

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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params
        params = [{'lags': 2, 'trend': 'c', 'order': 0},
                  {'lags': 1, 'trend': 'ct', 'order': 2},
                  {'auto_ardl': True, 'maxlag': 2, 'maxorder': 2}]
        return params

if __name__ =='__main__':
    from sktime.utils.estimator_checks import check_estimator
    from sktime.registry._lookup import all_tags
    from sktime.tests.test_all_estimators import TestAllEstimators, QuickTester
    from statsmodels.datasets import longley, grunfeld
    from sktime.forecasting.base import ForecastingHorizon
    from numpy.testing import assert_allclose
    #print(TestAllEstimators().run_tests(ARDL))
    #print(all_tags('forecaster'))
    #tsa = TestAllEstimators()
    #tsa.test_fit_does_not_overwrite_hyper_params(ARDL)
    print(check_estimator(ARDL, fixtures_to_run='test_fit_does_not_overwrite_hyper_params[ARDL-1-ForecasterFitPredictMultivariateNoX]', return_exceptions=False))
    #print(check_estimator(ARDL))

    def test_against_statsmodels():
        """
        Compares sktime's ARDL interface with statsmodels ARDL
        """
        # data
        data = longley.load_pandas().data
        oos = data.iloc[-5:, :]
        data = data.iloc[:-5, :]
        y = data.TOTEMP
        X = None
        X_oos = None
        # fit
        sm_ardl = _ARDL(y, 2, X, trend="c")
        res = sm_ardl.fit()
        ardl_sktime = ARDL(lags=2, trend='c')
        ardl_sktime.fit(y=y, X=X, fh=None)
        # predict
        fh = ForecastingHorizon([1, 2, 3])
        start, end = y.shape[0] + fh[0] - 1, y.shape[0] + fh[-1] - 1
        y_pred_stats = sm_ardl.predict(res.params, start=start, end=end, exog_oos=X_oos)
        y_pred = ardl_sktime.predict(fh=fh, X=X_oos)
        print(y_pred)
        print(y_pred_stats)
        return assert_allclose(y_pred, y_pred_stats)
    #print(test_against_statsmodels())

