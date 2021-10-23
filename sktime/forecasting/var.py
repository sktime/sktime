# -*- coding: utf-8 -*-
"""Vector Auto Regressor."""
__all__ = ["VAR"]
__author__ = ["thayeylolu", "aiwalter"]

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR as _VAR

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base.adapters import _StatsModelsAdapter


class VAR(_StatsModelsAdapter):
    """
    A VAR model is a generalisation of the univariate autoregressive.

    A model for forecasting a vector of time series[1].

    Parameters
    ----------
    maxlags: int or None (default= None)
        Maximum number of lags to check for order selection,
        defaults to 12 * (nobs/100.)**(1./4)
    method : str
        Estimation method to use
    verbose : bool (default = False)
        Print order selection output to the screen
    missing: str, optional (default='none')
        A string specifying if data is missing

    References
    ----------
    [1] Athanasopoulos, G., Poskitt, D. S., & Vahid, F. (2012).
    Two canonical VARMA forms: Scalar component models vis-à-vis the echelon form.
    Econometric Reviews, 31(1), 60–83, 2012.

    Examples
    --------
    >>> from sktime.forecasting.var import VAR
    >>> from sktime.datasets import load_longley
    >>> _, y = load_longley()
    >>> forecaster = VAR()
    >>> forecaster.fit(y)
    VAR(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])
    """

    _fitted_param_names = ("aic", "fpe", "hqic", "bic")

    _tags = {
        "scitype:y": "multivariate",
        "y_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "univariate-only": False,
        "ignores-exogeneous-X": False,
    }

    def __init__(
        self,
        maxlags=None,
        method="ols",
        verbose=False,
        trend="c",
        missing="none",
    ):
        # Model params
        self.trend = trend
        self.maxlags = maxlags
        self.method = method
        self.verbose = verbose
        self.missing = missing

        super(VAR, self).__init__()

    def _fit_forecaster(self, y, X=None):
        """Fit forecaster to training data.

        Wraps Statsmodel's VAR fit method.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        self : returns an instance of self.
        """
        self._forecaster = _VAR(endog=y, exog=X, missing=self.missing)
        self._fitted_forecaster = self._forecaster.fit(
            trend=self.trend,
            maxlags=self.maxlags,
            method=self.method,
            verbose=self.verbose,
        )
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Wrap Statmodel's VAR forecast method.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasters horizon with the steps ahead to to predict.
            Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.
        return_pred_int : bool, optional (default=False)
        alpha : int or list, optional (default=0.95)

        Returns
        -------
        y_pred : np.ndarray
            Returns series of predicted values.
        """
        y_pred_outsample = None
        y_pred_insample = None
        exog_future = X.values if X is not None else None
        # fh in stats
        # fh_int = fh.to_absolute_int(self._y.index[0], self._y.index[-1])
        fh_int = fh.to_relative(self.cutoff)
        n_lags = self._fitted_forecaster.k_ar

        # out-sample predictions
        if fh_int.max() > 0:
            y_pred_outsample = self._fitted_forecaster.forecast(
                y=self._y.values[-n_lags:],
                steps=fh_int[-1],
                exog_future=exog_future,
            )
        # in-sample prediction by means of residuals
        if fh_int.min() <= 0:
            y_pred_insample = self._y - self._fitted_forecaster.resid
            y_pred_insample = y_pred_insample.values

        if y_pred_insample is not None and y_pred_outsample is not None:
            y_pred = np.concatenate([y_pred_outsample, y_pred_insample], axis=0)
        else:
            y_pred = (
                y_pred_insample if y_pred_insample is not None else y_pred_outsample
            )

        index = fh.to_absolute(self.cutoff)
        index.name = self._y.index.name
        y_pred = pd.DataFrame(
            y_pred[fh.to_indexer(self.cutoff), :],
            index=fh.to_absolute(self.cutoff),
            columns=self._y.columns,
        )
        return y_pred

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict
        """
        params = {"maxlags": 3}
        return params
