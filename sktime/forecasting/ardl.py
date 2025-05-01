# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements ARDL Model as interface to statsmodels."""

import warnings

import pandas as pd

from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base.adapters import _StatsModelsAdapter
from sktime.forecasting.base.adapters._statsmodels import _coerce_int_to_range_index

_all_ = ["ARDL"]
__author__ = ["kcc-lion"]


class ARDL(_StatsModelsAdapter):
    """Autoregressive Distributed Lag (ARDL) Model.

    Direct interface for statsmodels.tsa.ardl.ARDL

    Parameters
    ----------
    lags : {int, list[int]}, optional
        Only considered if auto_ardl is False
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
    order : {int, sequence[int], dict}, optional
        Only considered if auto_ardl is False
        If int, uses lags 0, 1, ..., order  for all exog variables. If
        sequence[int], uses the ``order`` for all variables. If a dict,
        applies the lags series by series. If ``exog`` is anything other
        than a DataFrame, the keys are the column index of exog (e.g., 0,
        1, ...). If a DataFrame, keys are column names.
    fixed : array_like, optional
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
    cov_type : str, optional
        The covariance estimator to use. The most common choices are listed
        below.  Supports all covariance estimators that are available
        in ``OLS.fit``.

        * 'nonrobust' - The class OLS covariance estimator that assumes
          homoskedasticity.
        * 'HC0', 'HC1', 'HC2', 'HC3' - Variants of White's
          (or Eiker-Huber-White) covariance estimator. ``HC0`` is the
          standard implementation.  The other make corrections to improve
          the finite sample performance of the heteroskedasticity robust
          covariance estimator.
        * 'HAC' - Heteroskedasticity-autocorrelation robust covariance
          estimation. Supports cov_kwds.

          - ``maxlags`` integer (required) : number of lags to use.
          - ``kernel`` callable or str (optional) : kernel
              currently available kernels are ['bartlett', 'uniform'],
              default is Bartlett.
          - ``use_correction`` bool (optional) : If true, use small sample
              correction.
    cov_kwds : dict, optional
        A dictionary of keyword arguments to pass to the covariance
        estimator. ``nonrobust`` and ``HC#`` do not support cov_kwds.
    use_t : bool, optional
        A flag indicating that inference should use the Student's t
        distribution that accounts for model degree of freedom.  If False,
        uses the normal distribution. If None, defers the choice to
        the cov_type. It also removes degree of freedom corrections from
        the covariance estimator when cov_type is 'nonrobust'.
    auto_ardl : bool, optional
        A flag indicating whether the number of lags should be determined automatically.
    maxlag : int, optional
        Only considered if auto_ardl is True.
        The maximum lag to consider for the endogenous variable.
    maxorder : {int, dict}
        Only considered if auto_ardl is True.
        If int, sets a common max lag length for all exog variables. If
        a dict, then sets individual lag length. They keys are column names
        if exog is a DataFrame or column indices otherwise.
    ic : {"aic", "bic", "hqic"}, optional
        Only considered if auto_ardl is True.
        The information criterion to use in model selection.
    glob : bool, optional
        Only considered if auto_ardl is True.
        Whether to consider all possible submodels of the largest model
        or only if smaller order lags must be included if larger order
        lags are.  If ``True``, the number of model considered is of the
        order 2**(maxlag + k * maxorder) assuming maxorder is an int. This
        can be very large unless k and maxorder are bot relatively small.
        If False, the number of model considered is of the order
        maxlag*maxorder**k which may also be substantial when k and maxorder
        are large.
    X_oos : array_like, optional
        An array containing out-of-sample values of the exogenous
        variables. Must have the same number of columns as the X
        and at least as many rows as the number of out-of-sample forecasts.
    fixed_oos : array_like, optional
        An array containing out-of-sample values of the fixed variables.
        Must have the same number of columns as the fixed array
        and at least as many rows as the number of out-of-sample forecasts.
    dynamic : {bool, int, str, datetime, Timestamp}, optional
        Integer offset relative to ``start`` at which to begin dynamic
        prediction. Prior to this observation, true endogenous values
        will be used for prediction; starting with this observation and
        continuing through the end of prediction, forecasted endogenous
        values will be used instead. Datetime-like objects are not
        interpreted as offsets. They are instead used to find the index
        location of ``dynamic`` which is then used to to compute the offset.

    Notes
    -----
    The full specification of an ARDL is

    .. math ::

       Y_t = delta_0 + delta_1 t + delta_2 t^2
             + sum_{i=1}^{s-1} gamma_i I_{[(mod(t,s) + 1) = i]}
             + sum_{j=1}^p phi_j Y_{t-j}
             + sum_{l=1}^k sum_{m=0}^{o_l} beta_{l,m} X_{l, t-m}
             + Z_t lambda
             + epsilon_t

    where :math:`delta_bullet` capture trends, :math:`gamma_bullet`
    capture seasonal shifts, s is the period of the seasonality, p is the
    lag length of the endogenous variable, k is the number of exogenous
    variables :math:`X_{l}`, :math:`o_l` is included the lag length of
    :math:`X_{l}`, :math:`Z_t` are ``r`` included fixed regressors and
    :math:`epsilon_t` is a white noise shock. If ``causal`` is ``True``,
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
    Use ARDL on macroeconomic data
    >>> from sktime.datasets import load_macroeconomic
    >>> from sktime.forecasting.ardl import ARDL
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> data = load_macroeconomic()  # doctest: +SKIP
    >>> oos = data.iloc[-5:, :]  # doctest: +SKIP
    >>> data = data.iloc[:-5, :]  # doctest: +SKIP
    >>> y = data.realgdp  # doctest: +SKIP
    >>> X = data[["realcons", "realinv"]]  # doctest: +SKIP
    >>> X_oos = oos[["realcons", "realinv"]]  # doctest: +SKIP
    >>> ardl = ARDL(lags=2, order={"realcons": 1, "realinv": 2}, trend="c")\
    # doctest: +SKIP
    >>> ardl.fit(y=y, X=X)  # doctest: +SKIP
    ARDL(lags=2, order={'realcons': 1, 'realinv': 2})
    >>> fh = ForecastingHorizon([1, 2, 3])  # doctest: +SKIP
    >>> y_pred = ardl.predict(fh=fh, X=X_oos)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["bashtage", "kcc-lion"],
        # bashtage for statsmodels ARDL
        "maintainers": "kcc-lion",
        "python_dependencies": "statsmodels>=0.13.0",
        # estimator type
        # --------------
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "capability:missing_values": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
        "X-y-must-have-same-index": True,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "capability:pred_int": False,  # does forecaster implement proba forecasts?
    }

    def __init__(
        self,
        lags=None,
        order=None,
        fixed=None,
        causal=False,
        trend="c",
        seasonal=False,
        deterministic=None,
        hold_back=None,
        period=None,
        missing="none",
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=True,
        auto_ardl=False,
        maxlag=None,
        maxorder=None,
        ic="bic",
        glob=False,
        fixed_oos=None,
        X_oos=None,
        dynamic=False,
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
            raise ValueError("lags should not be specified if auto_ardl is True")

        super().__init__()

    def check_param_validity(self, X):
        """Check for the validity of entered parameter combination."""
        inner_order = self.order
        inner_auto_ardl = self.auto_ardl

        if not self.auto_ardl:
            if inner_order is not None and not isinstance(X, pd.DataFrame):
                inner_order = 0
                warnings.warn(
                    "X is none but the order for the exogenous variables was"
                    " specified. Order was thus set to 0",
                    stacklevel=2,
                )
        else:
            if not isinstance(X, pd.DataFrame):
                inner_order = 0
                inner_auto_ardl = False
                warnings.warn(
                    "X is none but auto_ardl was set to True. auto_ardl was"
                    " thus set to False with order=0",
                    stacklevel=2,
                )
        return inner_order, inner_auto_ardl

    # todo: implement this, mandatory
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
        from statsmodels.tsa.ardl import ARDL as _ARDL
        from statsmodels.tsa.ardl import ardl_select_order as _ardl_select_order

        # statsmodels does not support the pd.Int64Index as required,
        # so we coerce them here to pd.RangeIndex
        if isinstance(y, pd.Series) and pd.api.types.is_integer_dtype(y.index):
            y, X = _coerce_int_to_range_index(y, X)

        # validity check of passed params
        # certain parameter combinations (e.g. (1) order != 0 and X=None,
        # (2) auto_ardl=True and X=None) cause errors
        # Thus, the below function checks for validity of params,
        # resets them appropriately and issues a  warning if need be
        inner_order, inner_auto_ardl = self.check_param_validity(X)

        if not inner_auto_ardl:
            self._forecaster = _ARDL(
                endog=y,
                lags=self.lags,
                exog=X,
                order=inner_order,
                trend=self.trend,
                fixed=self.fixed,
                causal=self.causal,
                seasonal=self.seasonal,
                deterministic=self.deterministic,
                hold_back=self.hold_back,
                period=self.period,
                missing=self.missing,
            )

            self._fitted_forecaster = self._forecaster.fit(
                cov_type=self.cov_type, cov_kwds=self.cov_kwds, use_t=self.use_t
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
                missing=self.missing,
            )

            self._fitted_forecaster = self._forecaster.model.fit(
                cov_type=self.cov_type, cov_kwds=self.cov_kwds, use_t=self.use_t
            )
        return self

    def summary(self):
        """Get a summary of the fitted forecaster."""
        self.check_is_fitted()
        return self._fitted_forecaster.summary()

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
        # statsmodels forecasts all periods from start to end of forecasting
        # horizon, but only return given time points in forecasting horizon
        valid_indices = fh.to_absolute_index(self.cutoff)

        y_pred = self._fitted_forecaster.predict(
            start=start, end=end, exog=self._X, exog_oos=X, fixed_oos=self.fixed_oos
        )
        y_pred.name = self._y.name
        return y_pred.loc[valid_indices]

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
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        warnings.warn("Defaulting to `update_params=True`", stacklevel=2)
        update_params = True
        if update_params:
            # default to re-fitting if update is not implemented
            warnings.warn(
                f"NotImplementedWarning: {self.__class__.__name__} "
                f"does not have a custom `update` method implemented. "
                f"{self.__class__.__name__} will be refit each time "
                f"`update` is called with update_params=True.",
                stacklevel=2,
            )
            # we need to overwrite the mtype last seen, since the _y
            #    may have been converted
            mtype_last_seen = self._y_mtype_last_seen
            # refit with updated data, not only passed data
            self.fit(y=self._y, X=self._X, fh=self._fh)
            # todo: should probably be self._fit, not self.fit
            # but looping to self.fit for now to avoid interface break
            self._y_mtype_last_seen = mtype_last_seen

        # if update_params=False, and there are no components, do nothing
        # if update_params=False, and there are components, we update cutoffs
        elif self.is_composite():
            # default to calling component _updates if update is not implemented
            warnings.warn(
                f"NotImplementedWarning: {self.__class__.__name__} "
                f"does not have a custom `update` method implemented. "
                f"{self.__class__.__name__} will update all component cutoffs each time"
                f" `update` is called with update_params=False.",
                stacklevel=2,
            )
            comp_forecasters = self._components(base_class=BaseForecaster)
            for comp in comp_forecasters.values():
                comp.update(y=y, X=X, update_params=False)

        return self

    def _get_fitted_params(self):
        """Get fitted parameters.

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        from statsmodels.tsa.ardl import ARDL as _ARDL

        fitted_params = {}
        if isinstance(self._forecaster, _ARDL):
            fitted_params["score"] = self._forecaster.score(
                self._fitted_forecaster.params
            )
            fitted_params["hessian"] = self._forecaster.hessian(
                self._fitted_forecaster.params
            )
            fitted_params["information"] = self._forecaster.information(
                self._fitted_forecaster.params
            )
            fitted_params["loglike"] = self._forecaster.loglike(
                self._fitted_forecaster.params
            )
        else:
            if self._X is not None:
                fitted_params["score"] = self._fitted_forecaster.model.score(
                    self._fitted_forecaster.params
                )
                for x in ["_aic", "_bic", "_hqic"]:
                    fitted_params[x[1:]] = eval("self._forecaster." + x)
                fitted_params["hessian"] = self._fitted_forecaster.model.hessian(
                    self._fitted_forecaster.params
                )
                fitted_params["information"] = (
                    self._fitted_forecaster.model.information(
                        self._fitted_forecaster.params
                    )
                )
                fitted_params["loglike"] = self._fitted_forecaster.model.loglike(
                    self._fitted_forecaster.params
                )
        return fitted_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params = [
            {"lags": 1, "trend": "c", "order": 2},
            {"lags": 1, "trend": "ct"},
            {"auto_ardl": True, "maxlag": 2, "maxorder": 2},
        ]
        return params
