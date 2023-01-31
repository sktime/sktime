# -*- coding: utf-8 -*-
# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements AutoReg."""

__all__ = ["AutoReg"]
__author__ = ["pranavvp16"]

from sktime.forecasting.base.adapters import _StatsModelsAdapter


class AutoReg(_StatsModelsAdapter):
    """AutoReg forecaster.

    Direct interface for 'statsmodels.tsa.ar_model.AutoReg'

    Parameters
    ----------
    lags : {None,int,list[int]},default=None
        The number of lags to include in the model if an integer or the list
        of lag indices to include. For example, [1, 4] will only include lags
        1 and 4 while lags=4 will include lags 1, 2, 3, and 4. None excludes
        all AR lags, and behave identically to 0.
    trend : {'n','c','t','ct'},default='c'
        The trend to include in the model:
        ‘n’ - No trend.
        ‘c’ - Constant only.
        ‘t’ - Time trend only.
        ‘ct’ - Constant and time trend.
    seasonal : bool,default=False
        Flag indicating whether to include seasonal dummies in the model. If
        seasonal is True and trend includes ‘c’, then the first period is
        excluded from the seasonal terms.
    hold_back : {None,int},default=None
        Initial observations to exclude from the estimation sample. If None, then
        hold_back is equal to the maximum lag in the model. Set to a non-zero
        value to produce comparable models with different lag length. For example,
        to compare the fit of a model with lags=3 and lags=1, set hold_back=3 which
        ensures that both models are estimated using observations 3,…,nobs. hold_back
        must be >= the maximum lag in the model.
    period : {None,int},default=None
        The period of the data. Only used if seasonal is True. This parameter can be
        omitted if using a pandas object for endog that contains a recognized frequency.
    missing : str,default='none'
        Available options are ‘none’, ‘drop’, and ‘raise’. If ‘none’, no nan checking is
        done. If ‘drop’, any observations with nans are dropped. If ‘raise’, an error is
        raised. Default is ‘none’.
    deterministic : DeterministicProcess,default=None
        A deterministic process. If provided, trend and seasonal are ignored. A warning
        is raised if trend is not “n” and seasonal is not False.
    old_names : bool,default=False
        Flag indicating whether to use the v0.11 names or the v0.12+ names.

    References
    ----------
    [1] Athanasopoulos, G., Poskitt, D. S., & Vahid, F. (2012).
    Two canonical VARMA forms: Scalar component models vis-à-vis the echelon form.
    Econometric Reviews, 31(1), 60–83, 2012.

    Examples
    --------
    >>> from sktime.forecasting.autoreg import AutoReg
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = AutoReg()
    >>> forecaster.fit(y)
    AutoReg(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
 """
    _tags = {
        "scitype:y":"univariate",
        "ignores-exogeneous-X": False,
    }

    def __init__(
        self,
        lags=None,
        trend='c',
        seasonal=False,
        hold_back=None,
        period=None,
        missing='none',
        deterministic=None,
        old_names=False
     ):

        self.lags = lags
        self.trend = trend
        self.seasonal = seasonal
        self.hold_back = hold_back
        self.period = period
        self.missing = missing
        self.deterministic = deterministic
        self.old_names = old_names

        super(AutoReg,self).__init__()

    def _fit_forecaster(self, y_train, X_train=None):
        from statsmodels.tsa.api import AutoReg as _AutoReg

        self._forecaster = _AutoReg(
            endog=y_train,
            exog=X_train,
            lags=self.lags,
            trend=self.trend,
            seasonal=self.seasonal,
            hold_back=self.hold_back,
            period=self.period,
            missing=self.missing,
            deterministic=self.deterministic,
            old_names=self.old_names
        )
        self._fitted_forecaster = self._forecaster.fit()
        """Get a summary of the fitted forecaster.
           This is the same as the implementation in statsmodels:
           https://www.statsmodels.org/dev/examples/notebooks/generated/autoregressions.html               
            """
        return self._fitted_forecaster.summary()



