"""
Implements a residual boosting forecaster.

Which is an easy way to turn a forecaster without exogenous capability into one with.
"""

# copyright: sktime developers, BSD-3-Clause

__all__ = ["ResidualBoostingForecaster"]
__author__ = ["Sanchay117"]

from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster


class ResidualBoostingForecaster(BaseForecaster):
    """Residual boosting: add exogenous support to any base forecaster.

    Parameters
    ----------
    base_forecaster : sktime forecaster
        Point-forecast model that may ignore X.
    residual_forecaster : sktime forecaster
        Model trained on the base model's in-sample residuals.
    """

    _tags = {
        "authors": ["Sanchay117"],
    }

    def __init__(self, base_forecaster, residual_forecaster):
        self.base_forecaster = base_forecaster
        self.residual_forecaster = residual_forecaster
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """
        Fit base forecaster and residual forecaster.

        1. Fit clone A of base_forecaster to X, y, and compute in-sample
           forecast residuals r
        2. Fit clone B of base_forecaster to X, y, with fh
        3. Fit clone of residual_forecaster to X, r
        """
        # 1) base clone A for residual extraction
        base_forecaster_copy = clone(self.base_forecaster)
        base_forecaster_copy.fit(y=y, X=X, fh=None)
        y_pred = base_forecaster_copy.predict_in_sample()
        residuals = y - y_pred

        # 2) base clone B for future prediction
        self.base_forecaster_copy_fh = clone(self.base_forecaster)
        self.base_forecaster_copy_fh.fit(y=y, X=X, fh=fh)

        # 3) residual forecaster on residuals
        self.residual_forecaster_copy = clone(self.residual_forecaster)
        self.residual_forecaster_copy.fit(y=residuals, X=X, fh=fh)

        return self

    def _predict(self, fh, X=None):
        """
        Forecast = base forecast + residual forecast.

        1. Use clone B of base_forecaster to obtain a prediction y_pred_base
        2. Use residual_forecaster clone to obtain a prediction y_pred_resid
        3. Return y_pred_base + y_pred_resid
        """
        y_base = self.base_forecaster_copy_fh.predict(fh=fh, X=X)
        y_resid = self.residual_forecaster_copy.predict(fh=fh, X=X)
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
        from sktime.forecasting.naive import NaiveForecaster

        params1 = {
            "base_forecaster": NaiveForecaster(strategy="last"),
            "residual_forecaster": NaiveForecaster(strategy="mean"),
        }
        params2 = {
            "base_forecaster": NaiveForecaster(strategy="drift"),
            "residual_forecaster": NaiveForecaster(strategy="last"),
        }
        return [params1, params2]
