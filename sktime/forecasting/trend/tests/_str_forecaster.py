"""Implements STRForecaster based on statsmodels."""

import pandas as pd
from sktime.forecasting.base import BaseForecaster

class STRForecaster(BaseForecaster):
    """STRForecaster using R's STR (Seasonal Trend using Regression) implementation."""
    
    _tags = {
        "authors": ["tensorflow-as-tf", "mloning", "aiwalter", "fkiraly"],
        "maintainers": ["tensorflow-as-tf"],
        "scitype:y": "univariate",
        "ignores-exogeneous-X": False,
        "handles-missing-data": False,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "python_dependencies": "rpy2",
    }

    def __init__(
        self,
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
        inner_iter=None,
        outer_iter=None,
        forecaster_trend=None,
        forecaster_seasonal=None,
        forecaster_resid=None,
    ):
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
        self.inner_iter = inner_iter
        self.outer_iter = outer_iter
        self.forecaster_trend = forecaster_trend
        self.forecaster_seasonal = forecaster_seasonal
        self.forecaster_resid = forecaster_resid
        super().__init__()

        # Check if forecasters use exogenous variables
        for forecaster in (
            self.forecaster_trend,
            self.forecaster_seasonal,
            self.forecaster_resid,
        ):
            if forecaster is not None and not forecaster.get_tag(
                "ignores-exogeneous-X"
            ):
                ignore_exogenous = False
                break
        else:
            ignore_exogenous = True

        self.set_tags(**{"ignores-exogeneous-X": ignore_exogenous})

    def _fit(self, y, X, fh):
        """Fit forecaster to training data."""
        from rpy2.robjects import r, pandas2ri
        from rpy2.robjects.packages import importr

        # Import R's forecast package
        forecast = importr('forecast')
        pandas2ri.activate()

        # Convert pandas series to R time series
        r_ts = pandas2ri.py2rpy(y)
        r(f'ts_data <- ts({r_ts}, frequency={self.sp})')
        
        # Apply STR decomposition
        r('str_decomp <- stR(ts_data)')
        
        # Extract components
        self.seasonal_ = pd.Series(r('str_decomp$seasonal'), index=y.index)
        self.resid_ = pd.Series(r('str_decomp$remainder'), index=y.index)
        self.trend_ = pd.Series(r('str_decomp$trend'), index=y.index)

        # Set up forecasters
        from sktime.forecasting.naive import NaiveForecaster
        
        self.forecaster_seasonal_ = (
            NaiveForecaster(sp=self.sp, strategy="last")
            if self.forecaster_seasonal is None
            else self.forecaster_seasonal.clone()
        )
        self.forecaster_trend_ = (
            NaiveForecaster(strategy="drift")
            if self.forecaster_trend is None
            else self.forecaster_trend.clone()
        )
        self.forecaster_resid_ = (
            NaiveForecaster(sp=self.sp, strategy="mean")
            if self.forecaster_resid is None
            else self.forecaster_resid.clone()
        )

        # Fit forecasters to components
        self.forecaster_seasonal_.fit(y=self.seasonal_, X=X, fh=fh)
        self.forecaster_trend_.fit(y=self.trend_, X=X, fh=fh)
        self.forecaster_resid_.fit(y=self.resid_, X=X, fh=fh)

    def _predict(self, fh, X):
        """Forecast time series at future horizon."""
        y_pred_seasonal = self.forecaster_seasonal_.predict(fh=fh, X=X)
        y_pred_trend = self.forecaster_trend_.predict(fh=fh, X=X)
        y_pred_resid = self.forecaster_resid_.predict(fh=fh, X=X)
        y_pred = y_pred_seasonal + y_pred_trend + y_pred_resid
        y_pred.name = self._y.name
        return y_pred

    def _update(self, y, X=None, update_params=True):
        """Update model with new data."""
        from rpy2.robjects import r, pandas2ri
        from rpy2.robjects.packages import importr

        forecast = importr('forecast')
        pandas2ri.activate()

        r_ts = pandas2ri.py2rpy(y)
        r(f'ts_data <- ts({r_ts}, frequency={self.sp})')
        r('str_decomp <- stR(ts_data)')

        self.seasonal_ = pd.Series(r('str_decomp$seasonal'), index=y.index)
        self.resid_ = pd.Series(r('str_decomp$remainder'), index=y.index)
        self.trend_ = pd.Series(r('str_decomp$trend'), index=y.index)

        self.forecaster_seasonal_.update(y=self.seasonal_, X=X, update_params=update_params)
        self.forecaster_trend_.update(y=self.trend_, X=X, update_params=update_params)
        self.forecaster_resid_.update(y=self.resid_, X=X, update_params=update_params)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sktime.forecasting.naive import NaiveForecaster

        params_list = [
            {},
            {
                "sp": 3,
                "seasonal": 7,
                "trend": 5,
                "seasonal_deg": 2,
                "trend_deg": 2,
                "robust": True,
                "seasonal_jump": 2,
                "trend_jump": 2,
                "low_pass_jump": 2,
                "forecaster_trend": NaiveForecaster(strategy="drift"),
                "forecaster_seasonal": NaiveForecaster(sp=3),
                "forecaster_resid": NaiveForecaster(strategy="mean"),
            },
        ]
        return params_list