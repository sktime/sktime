# -*- coding: utf-8 -*-
"""Implements STLForecaster based on statsmodels.tsa.seasonal.STL implementation."""

__author__ = ["aiwalter"]
__all__ = ["STLForecaster"]


import pandas as pd
from sklearn.base import clone
from statsmodels.tsa.seasonal import STL as _STL

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.naive import NaiveForecaster


class STLForecaster(BaseForecaster):
    """STLForecaster.

    The STLForecaster is using an STL to decompose the given
    series y into the three components trend, season and residuals [1]_. Then,
    the trend_forecaster, seasonal_forecaster and resid_forecaster are fitted
    on the components individually to forecast them also individually. The
    final forecast is then the sum of the three component forecasts.

    Parameters
    ----------
    sp : int, optional
        Seasonal period for defaulting forecasters and/or STL,
        by default None. Can only be used if at least on the forecasters
        or the stl is None.
    trend_forecaster : sktime forecaster, optional
        Forecaster to be fitted on trend_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="drift") is used.
    seasonal_forecaster : sktime forecaster, optional
        Forecaster to be fitted on seasonal_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="last") is used.
    resid_forecaster : sktime forecaster, optional
        Forecaster to be fitted on resid_ component of the
        STL, by default None. If None, then
        a NaiveForecaster(strategy="mean") is used.

    Attributes
    ----------
    trend_ : pd.Series
        Trend component.
    seasonal_ : pd.Series
        Seasonal component.
    resid_ : pd.Series
        Residuals component.
    trend_forecaster_ : sktime forecaster
        Fitted trend forecaster.
    seasonal_forecaster_ : sktime forecaster
        Fitted seasonal forecaster.
    resid_forecaster_ : sktime forecaster
        Fitted residual forecaster.
    stl_ : STL
        Fitted statsmodels.STL

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.stl import STLForecaster
    >>> y = load_airline()
    >>> forecaster = STLForecaster(sp=12)
    >>> forecaster.fit(y)
    STLForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])

    See Also
    --------
    STL

    References
    ----------
    .. [1] R. B. Cleveland, W. S. Cleveland, J.E. McRae, and I. Terpenning (1990)
       STL: A Seasonal-Trend Decomposition Procedure Based on LOESS.
       Journal of Official Statistics, 6, 3-73.
    """

    _tags = {
        "scitype:y": "univariate",  # which y are fine? univariate/multivariate/both
        "ignores-exogeneous-X": False,  # does estimator ignore the exogeneous X?
        "handles-missing-data": False,  # can estimator handle missing data?
        "y_inner_mtype": "pd.Series",  # which types do _fit, _predict, assume for y?
        "X_inner_mtype": "pd.DataFrame",  # which types do _fit, _predict, assume for X?
        "requires-fh-in-fit": False,  # is forecasting horizon already required in fit?
    }

    def __init__(
        self,
        trend_forecaster=None,
        seasonal_forecaster=None,
        resid_forecaster=None,
        sp=2,
        seasonal=7,
        trend=None,
        low_pass=None,
        seasonal_deg=1,
        trend_deg=1,
        low_pass_deg=1,
        robust=False,
        seasonal_jump=1,
        trend_jump=1,
        low_pass_jump=1,
    ):
        self.trend_forecaster = trend_forecaster
        self.seasonal_forecaster = seasonal_forecaster
        self.resid_forecaster = resid_forecaster
        self.sp = sp
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump
        super(STLForecaster, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)

        Returns
        -------
        self : returns an instance of self.
        """
        self.stl_ = _STL(
            y.values,
            period=self.sp,
            seasonal=self.seasonal,
            trend=self.trend,
            low_pass=self.low_pass,
            seasonal_deg=self.seasonal_deg,
            trend_deg=self.trend_deg,
            low_pass_deg=self.low_pass_deg,
            robust=self.robust,
            seasonal_jump=self.seasonal_jump,
            trend_jump=self.trend_jump,
            low_pass_jump=self.low_pass_jump,
        ).fit()

        self.seasonal_ = pd.Series(self.stl_.seasonal, index=y.index)
        self.resid_ = pd.Series(self.stl_.resid, index=y.index)
        self.trend_ = pd.Series(self.stl_.trend, index=y.index)

        # setting defualt forecasters if required
        self.seasonal_forecaster_ = (
            NaiveForecaster(sp=self.sp, strategy="last")
            if self.seasonal_forecaster is None
            else clone(self.seasonal_forecaster)
        )
        self.trend_forecaster_ = (
            NaiveForecaster(strategy="drift")
            if self.trend_forecaster is None
            else clone(self.trend_forecaster)
        )
        self.resid_forecaster_ = (
            NaiveForecaster(sp=self.sp, strategy="mean")
            if self.resid_forecaster is None
            else clone(self.resid_forecaster)
        )

        # fitting forecasters to different components
        self.seasonal_forecaster_.fit(y=self.seasonal_, X=X, fh=fh)
        self.trend_forecaster_.fit(y=self.trend_, X=X, fh=fh)
        self.resid_forecaster_.fit(y=self.resid_, X=X, fh=fh)

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon
        X : pd.DataFrame, optional (default=None)
                Exogenous time series
        return_pred_int : bool, optional (default=False)
            If True, returns prediction intervals for given alpha values.
        alpha : float or list, optional (default=0.95)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        y_pred_seasonal = self.seasonal_forecaster_.predict(fh=fh, X=X)
        y_pred_trend = self.trend_forecaster_.predict(fh=fh, X=X)
        y_pred_resid = self.resid_forecaster_.predict(fh=fh, X=X)
        y_pred = y_pred_seasonal + y_pred_trend + y_pred_resid
        return y_pred
