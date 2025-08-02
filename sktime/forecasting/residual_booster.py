"""
Implements a residual boosting forecaster.

Which is an easy way to turn a forecaster without exogenous capability into one with.
"""

# copyright: sktime developers, BSD-3-Clause

__all__ = ["ResidualBoostingForecaster"]
__author__ = ["Sanchay117", "felipeangelimvieira"]

import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class ResidualBoostingForecaster(BaseForecaster):
    """
    Residual boosting lets an endogenous forecaster act as if it were exogenous.

    It is a two stage reduction: we fit an auxiliary model on the in-sample residuals
    of the base model and add its future predictions back to the base forecasts.

    Parameters
    ----------
    base_forecaster : sktime forecaster
        Point-forecast model that may ignore X.
    residual_forecaster : sktime forecaster
        Model trained on the base model's in-sample residuals.

    Example
    -------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.residual_booster import ResidualBoostingForecaster
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.compose import make_reduction
    >>> from sklearn.linear_model import LinearRegression
    >>> y, X = load_longley()
    >>> fh = [1, 2, 3]
    >>> base = NaiveForecaster(strategy="last")
    >>> resid = make_reduction(
    ...     estimator=LinearRegression(),
    ...     strategy="recursive",
    ...     window_length=3,
    ... )
    >>> booster = ResidualBoostingForecaster(base, resid).fit(y, X=X, fh=fh)
    >>> y_pred = booster.predict(fh, X=X)
    """

    _tags = {
        "authors": ["Sanchay117", "felipeangelimvieira"],
        "capability:pred_int": False,
        "ignores-exogeneous-X": True,
        "capability:missing_values": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
    }

    def __init__(self, base_forecaster, residual_forecaster):
        self.base_forecaster = base_forecaster
        self.residual_forecaster = residual_forecaster
        super().__init__()

        exog = self.base_forecaster.get_tag(
            "ignores-exogeneous-X"
        ) or self.residual_forecaster.get_tag("ignores-exogeneous-X")

        miss = self.base_forecaster.get_tag(
            "capability:missing_values"
        ) and self.residual_forecaster.get_tag("capability:missing_values")

        in_sample = self.base_forecaster.get_tag(
            "capability:insample"
        ) or self.residual_forecaster.get_tag("capability:insample")

        pred_int_insample = self.residual_forecaster.get_tag(
            "capability:pred_int:insample"
        )

        self.set_tags(
            **{
                "ignores-exogeneous-X": exog,
                "capability:missing_values": miss,
                "capability:insample": in_sample,
                "capability:pred_int:insample": pred_int_insample,
            }
        )

    def _fit(self, y, X=None, fh=None):
        """
        Fit base forecaster and residual forecaster.

        1. Fit clone A of base_forecaster to X, y, and compute in-sample
           forecast residuals r
        2. Fit clone B of base_forecaster to X, y, with fh
        3. Fit clone of residual_forecaster to X, r
        """
        # clone A: fit on (y,X) to obtain in-sample residuals
        # 1. in-sample residuals
        self.base_insample_ = clone(self.base_forecaster).fit(y, X, fh)

        # Forecast insample
        if isinstance(y.index, pd.MultiIndex):
            time_idx = y.index.get_level_values(-1).unique()
        else:
            time_idx = y.index
        insample_fh = ForecastingHorizon(time_idx, is_relative=False)

        insample_preds = self.base_insample_.predict(fh=insample_fh, X=X)

        residuals = y - insample_preds

        # clone B: fit a fresh copy that knows the final fh
        # 2. future base model with final fh
        self.base_future_ = clone(self.base_forecaster).fit(y, X, fh)

        # clone C: fit residual model on errors
        # 3. residual model
        self.residual_forecaster_ = clone(self.residual_forecaster).fit(
            residuals, X, fh
        )
        return self

    def _predict(self, fh=None, X=None):
        """
        Forecast = base forecast + residual forecast.

        1. Use clone B of base_forecaster to obtain a prediction y_pred_base
        2. Use residual_forecaster clone to obtain a prediction y_pred_resid
        3. Return y_pred_base + y_pred_resid
        """
        y_base = self.base_future_.predict(fh=fh, X=X)
        y_resid = self.residual_forecaster_.predict(fh=fh, X=X)
        return y_base + y_resid

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create test instances of the estimator.
        """
        from sktime.forecasting.arima import StatsModelsARIMA
        from sktime.forecasting.naive import NaiveForecaster

        params1 = {
            "base_forecaster": NaiveForecaster(strategy="last"),
            "residual_forecaster": NaiveForecaster(strategy="mean"),
        }

        params2 = {
            "base_forecaster": StatsModelsARIMA(order=(1, 0, 0)),
            "residual_forecaster": NaiveForecaster(strategy="mean"),
        }

        return [params1, params2]
