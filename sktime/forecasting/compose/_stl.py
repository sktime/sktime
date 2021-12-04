#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file).
"""Implements stl forecaster based on STLTransformer and three forecasters."""

__author__ = ["aiwalter", "RNKuhns"]
__all__ = ["STLForecaster"]


from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.naive import NaiveForecaster
from sktime.transformations.series.detrend import STLTransformer


class STLForecaster(BaseForecaster):
    """STLForecaster.

    Parameters
    ----------
    seasonal_forecaster : sktime forecaster
    trend_forecaster : sktime forecaster
    resid_forecaster : sktime forecaster
    stl : sktime.STLTransformer
        Transformer used to extract season, trend and resid components.
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
        seasonal_forecaster=None,
        trend_forecaster=None,
        resid_forecaster=None,
        stl=None,
        sp=None,
    ):
        self.seasonal_forecaster = seasonal_forecaster
        self.trend_forecaster = trend_forecaster
        self.resid_forecaster = resid_forecaster
        self.stl = stl
        self.sp = sp
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
        if (
            None
            not in [
                self.stl,
                self.seasonal_forecaster,
                self.trend_forecaster,
                self.resid_forecaster,
            ]
            and self.sp is not None
        ):
            raise ValueError(
                """
                The sp param can only be used if any of the forecasters or stl
                is None. Then sp is used to set the default sp for the forecaster(s)
                and/or STLTransformer.
                """
            )
        # set stl sp defautl if required
        _sp_stl = 2 if (self.sp is None and self.stl is None) else self.sp
        # set forecaster sp default if required
        _sp_forecaster = 1 if self.sp is None else self.sp
        # set default STLTransformer if required
        self.stl_ = STLTransformer(sp=_sp_stl) if self.stl is None else clone(self.stl)

        # fit STLTransformer and get fitted components
        self.stl_.fit(y)
        self.trend_ = self.stl_.trend_
        self.seasonal_ = self.stl_.seasonal_
        self.resid_ = self.stl_.resid_

        # setting defualt forecasters if required
        self.seasonal_forecaster_ = (
            NaiveForecaster(sp=_sp_forecaster, strategy="last")
            if self.seasonal_forecaster is None
            else clone(self.seasonal_forecaster)
        )
        self.trend_forecaster_ = (
            NaiveForecaster(strategy="drift")
            if self.trend_forecaster is None
            else clone(self.trend_forecaster)
        )
        self.resid_forecaster_ = (
            NaiveForecaster(sp=_sp_forecaster, strategy="mean")
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
