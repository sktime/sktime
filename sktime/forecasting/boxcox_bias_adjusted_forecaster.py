# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""BoxCoxBiasAdjustedForecaster implementation."""

__author__ = ["sanskarmodi8"]

import pandas as pd
from scipy.special import inv_boxcox

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.transformations.series.boxcox import BoxCoxTransformer


class BoxCoxBiasAdjustedForecaster(BaseForecaster):
    """Box-Cox Bias-Adjusted Forecaster.

    This module implements a forecaster that applies Box-Cox transformation
    and bias adjustment to the predictions of a wrapped forecaster.

    The bias adjustment is applied to both point predictions and prediction
    intervals to ensure consistent and accurate forecasts.

    Parameters
    ----------
    forecaster : BaseForecaster
        The wrapped forecaster to which Box-Cox transformation and
        bias adjustment will be applied.
    lmbda : float, optional (default=None)
        The Box-Cox transformation parameter. If None, it will be estimated.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) "Forecasting:
           principles and practice", 2nd edition, OTexts: Melbourne, Australia.
           OTexts.com/fpp2
    """

    def __init__(self, forecaster, lmbda=None):
        self.forecaster = forecaster
        self.lmbda = lmbda
        self.boxcox_transformer_ = None
        # Clone tags from the wrapped forecaster to this forecaster
        _DelegatedForecaster._set_delegated_tags(self, forecaster)
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to the training data.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self.boxcox_transformer_ = BoxCoxTransformer(lmbda=self.lmbda)
        self.y_transformed = self.boxcox_transformer_.fit_transform(y)

        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=self.y_transformed, X=X, fh=fh)

        # Check if the wrapped forecaster supports variance prediction
        if not hasattr(self.forecaster_, "predict_var"):
            raise ValueError(
                "The wrapped forecaster must implement a `predict_var` method "
                "to enable bias adjustment."
            )

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.

        Returns
        -------
        y_pred : pd.DataFrame
            Bias-adjusted point predictions.
        """
        y_pred_transformed = self.forecaster_.predict(fh, X)
        variance = self.forecaster_.predict_var(fh, X)

        y_pred = self._apply_bias_adjustment(y_pred_transformed, variance)

        return y_pred

    def _predict_interval(self, fh, X=None, coverage=None):
        """Compute prediction intervals for the forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        coverage : list of float, optional (default=None)
            Confidence levels for prediction intervals.

        Returns
        -------
        pred_int : pd.DataFrame
            Prediction intervals.
        """
        pred_int_transformed = self.forecaster_.predict_interval(fh, X, coverage)
        variance = self.forecaster_.predict_var(fh, X)

        lower_adjusted = self._apply_bias_adjustment(
            pred_int_transformed["lower"], variance
        )
        upper_adjusted = self._apply_bias_adjustment(
            pred_int_transformed["upper"], variance
        )

        return pd.concat([lower_adjusted, upper_adjusted], axis=1)

    def _apply_bias_adjustment(self, y, variance):
        """Apply bias adjustment for BoxCox Transformations.

        Parameters
        ----------
        y : pd.Series
            Transformed predictions or prediction intervals.
        variance : pd.Series
            Variance of the predictions.

        Returns
        -------
        y_adjusted : pd.Series
            Bias-adjusted predictions or prediction intervals.
        """
        lmbda = self.boxcox_transformer_.lmbda_
        w = self.y_transformed
        denominator = 2 * (lmbda * w + 1) ** 2
        adjustment = 1 + (variance * (1 - lmbda)) / denominator

        return inv_boxcox(y, lmbda) * adjustment

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster

        forecaster1 = NaiveForecaster(strategy="mean")
        forecaster2 = PolynomialTrendForecaster(degree=1)

        params1 = {"forecaster": forecaster1}
        params2 = {"forecaster": forecaster2, "lmbda": 0.5}

        return [params1, params2]
