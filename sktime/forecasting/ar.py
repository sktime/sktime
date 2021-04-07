#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

__author__ = ["Markus Löning"]
__all__ = ["AutoAR"]

from statsmodels.tsa.ar_model import ar_select_order

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class AutoAR(_StatsModelsAdapter):
    """
    Autor-egressive AR-X(p) model.
    Estimate an AR-X model using Conditional Maximum Likelihood (OLS)
    and automatically selecting the value for p.
    Parameters
    ----------
    maxlag : int, optional (default=None)
        The maximum lag to consider. If None, maxlag = 10 is used.
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
    hold_back : {None, int}
        Initial observations to exclude from the estimation sample.  If None,
        then hold_back is equal to the maximum lag in the model.  Set to a
        non-zero value to produce comparable models with different lag
        length.  For example, to compare the fit of a model with lags=3 and
        lags=1, set hold_back=3 which ensures that both models are estimated
        using observations 3,...,nobs. hold_back must be >= the maximum lag in
        the model.
    sp : {None, int}
        The seasonal periodicity of the data. Only used if seasonal is True. This
        parameter
        can be omitted if using a pandas object for y_train that contains a
        recognized frequency.
    missing : str
        Available options are 'none', 'drop', and 'raise'. If 'none', no nan
        checking is done. If 'drop', any observations with nans are dropped.
        If 'raise', an error is raised. Default is 'none'.
    ic : {'aic','bic','hqic','t-stat'}
        Criterion used for selecting the optimal lag length.
    old_names : bool
        Flag indicating whether to use the v0.11 names or the v0.12+ names.
        After v0.12 is released, the default names will change to the new
        names.
    References
    ----------
    ..[1] https://www.statsmodels.org/stable/_modules/statsmodels/tsa/ar_model.html
    """

    def __init__(
        self,
        maxlag,
        trend="c",
        seasonal=False,
        hold_back=None,
        sp=None,
        missing="none",
        ic="bic",
        glob=False,
        old_names=True,
    ):
        self.maxlag = maxlag
        self.trend = trend
        self.seasonal = seasonal
        self.hold_back = hold_back
        self.sp = sp
        self.missing = missing
        self.glob = glob
        self.ic = ic
        self.old_names = old_names

        super(AutoAR, self).__init__()

    def _fit_forecaster(self, y_train, X_train=None):
        self._fitted_forecaster = ar_select_order(
            y_train,
            maxlag=self.maxlag,
            ic=self.ic,
            glob=self.glob,
            trend=self.trend,
            seasonal=self.seasonal,
            exog=X_train,
            hold_back=self.hold_back,
            period=self.sp,
            missing=self.missing,
            old_names=self.old_names,
        ).model.fit()
        return self
