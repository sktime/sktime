# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Interfaces AutoReg Forecaster from statsmodels.

Available from statsmodels.tsa.ar_model.
"""

__author__ = ["jonathanbechtel", "mgazian000", "CTFallon"]
__all__ = ["AutoREG"]

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class AutoREG(_StatsModelsAdapter):
    """Autoregressive AR-X(p) model.

    Estimate an AR-X model using Conditional Maximum Likelihood (OLS).

    Parameters
    ----------
    endog : array_like
        A 1-d endogenous response variable. The dependent variable.
    lags : {None, int, list[int]}
        The number of lags to include in the model if an integer or the
        list of lag indices to include.  For example, [1, 4] will only
        include lags 1 and 4 while lags=4 will include lags 1, 2, 3, and 4.
        None excludes all AR lags, and behave identically to 0.
    trend : {'n', 'c', 't', 'ct'}
        The trend to include in the model:

        * 'n' - No trend.
        * 'c' - Constant only.
        * 't' - Time trend only.
        * 'ct' - Constant and time trend.

    seasonal : bool
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes 'c', then the first period
        is excluded from the seasonal terms.
    exog : array_like, optional
        Exogenous variables to include in the model. Must have the same number
        of observations as endog and should be aligned so that endog[i] is
        regressed on exog[i].
    hold_back : {None, int}
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    period : {None, int}
        The period of the data. Only used if seasonal is True. This parameter
        can be omitted if using a pandas object for endog that contains a
        recognized frequency.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.
    deterministic : DeterministicProcess
        A deterministic process.  If provided, trend and seasonal are ignored.
        A warning is raised if trend is not "n" and seasonal is not False.

    Examples
    --------
    Use AutoREG to forecast univariate data.

    >>> from sktime.forecasting.auto_reg import AutoREG  # doctest: +SKIP
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> data = load_airline()
    >>> autoreg_sktime = AutoREG(lags=2, trend="c")  # doctest: +SKIP
    >>> autoreg_sktime.fit(y=data)  # doctest: +SKIP
    AutoREG(lags=2)
    >>> fh = ForecastingHorizon([x for x in range(1, 13)])
    >>> y_pred = autoreg_sktime.predict(fh=fh)  # doctest: +SKIP


    Use AutoREG to forecast with exogenous data.

    >>> from sktime.forecasting.auto_reg import AutoREG  # doctest: +SKIP
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> y, X_og = load_longley()
    >>> X_oos = X_og.iloc[-5:, :]
    >>> y, X = y.iloc[:-5], X_og.iloc[:-5, :]
    >>> X, X_oos = X[["GNPDEFL", "GNP"]], X_oos[["GNPDEFL", "GNP"]]
    >>> autoreg_sktime = AutoREG(lags=2, trend="c")  # doctest: +SKIP
    >>> autoreg_sktime.fit(y=y, X=X)  # doctest: +SKIP
    AutoREG(lags=2)
    >>> fh = ForecastingHorizon([x for x in range(1, 4)])
    >>> y_pred = autoreg_sktime.predict(X=X_oos, fh=fh)  # doctest: +SKIP
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "python_version": None,
        "python_dependencies": "statsmodels>=0.13.0",
    }

    def __init__(
        self,
        lags=None,
        trend="c",
        seasonal=False,
        hold_back=None,
        period=None,
        missing="none",
        deterministic=None,
        cov_type="nonrobust",
        cov_kwds=None,
        use_t=True,
        dynamic=False,
    ):
        # Model Params
        self.lags = lags
        self.trend = trend
        self.seasonal = seasonal
        self.hold_back = hold_back
        self.period = period
        self.missing = missing
        self.deterministic = deterministic

        # Fit Params
        self.cov_type = cov_type
        self.cov_kwds = cov_kwds
        self.use_t = use_t

        # Predcit Params
        self.dynamic = dynamic

        # setup for get_fitted_params
        self._fitted_param_names = ("aic", "aicc", "bic", "hqic")

        super().__init__()

    def _fit_forecaster(self, y, X=None):
        """Fit forecaster to training data.

        private _fit_forecaster containing the core logic, called from fit

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
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        from statsmodels.tsa.ar_model import AutoReg as _AutoReg

        self._forecaster = _AutoReg(
            endog=y,
            lags=self.lags,
            trend=self.trend,
            seasonal=self.seasonal,
            exog=X,
            hold_back=self.hold_back,
            period=self.period,
            missing=self.missing,
            deterministic=self.deterministic,
        )

        self._fitted_forecaster = self._forecaster.fit(
            cov_type=self.cov_type, cov_kwds=self.cov_kwds, use_t=self.use_t
        )
        for param, value in self._fitted_forecaster.params.items():
            setattr(self, str(param) + "_", value)
            self._fitted_param_names = self._fitted_param_names + (str(param),)
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
        # statsmodels requires zero-based indexing starting at the
        # beginning of the training series when passing integers

        start, end = fh.to_absolute_int(self._y.index[0], self.cutoff)[[0, -1]]
        # statsmodels forecasts all periods from start to end of forecasting
        # horizon, but only return given time points in forecasting horizon
        valid_indices = fh.to_absolute_index(self.cutoff)

        y_pred = self._fitted_forecaster.predict(
            start=start, end=end, exog=self._X, exog_oos=X, dynamic=self.dynamic
        )
        y_pred.name = self._y.name

        return y_pred.loc[valid_indices]
        # implement here
        # IMPORTANT: avoid side effects to X, fh

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
        params = [
            {"lags": 2, "trend": "c"},
            {"lags": 2, "trend": "ct"},
            {"lags": 2, "trend": "t", "seasonal": True, "period": 2},
        ]
        return params
