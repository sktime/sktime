# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""BoxCoxBiasAdjustedForecaster implementation."""

__author__ = ["sanskarmodi8"]

import pandas as pd

from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.transformations.series.boxcox import BoxCoxTransformer


class BoxCoxBiasAdjustedForecaster(_DelegatedForecaster):
    """Box-Cox Bias-Adjusted Forecaster.

    This module implements a forecaster that applies Box-Cox transformation
    and bias adjustment to the predictions of a wrapped forecaster.

    The bias adjustment is implemented using Taylor series expansion of the
    method described in:
    Forecasting: Principles and Practice (2nd ed)
    Rob J Hyndman and George Athanasopoulos
    Monash University, Australia. OTexts.com/fpp2.

    For methods like `predict_proba`, the behavior of the wrapped forecaster
    is directly used without any additional adjustments. This means that
    probability estimates are provided as they are by the underlying forecaster.
    Users should ensure that the wrapped forecaster's `predict_proba` output
    is suitable for their needs, as no transformation or adjustment will be applied
    to these estimates by the `BoxCoxBiasAdjustedForecaster`.

    Parameters
    ----------
    forecaster : BaseForecaster
        The wrapped forecaster to which Box-Cox transformation and
        bias adjustment will be applied.
    lmbda : float, optional (default=None)
        The Box-Cox transformation parameter. If None, it will be estimated.
    """

    _delegate_name = "forecaster"

    def __init__(self, forecaster, lmbda=None):
        self.forecaster = forecaster
        self.lmbda = lmbda
        self.boxcox_transformer_ = None
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
        y_transformed = self.boxcox_transformer_.fit_transform(y)

        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=y_transformed, X=X, fh=fh)

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

        y_pred = self.boxcox_transformer_.inverse_transform(y_pred_transformed)
        y_adjusted = self._apply_bias_adjustment(y_pred, variance)

        return y_adjusted

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
        pred_int = self.boxcox_transformer_.inverse_transform(pred_int_transformed)

        variance = self.forecaster_.predict_var(fh, X)

        lower_adjusted = self._apply_bias_adjustment(pred_int["lower"], variance)
        upper_adjusted = self._apply_bias_adjustment(pred_int["upper"], variance)

        return pd.concat([lower_adjusted, upper_adjusted], axis=1)

    def _apply_bias_adjustment(self, y, variance):
        """Apply bias adjustment using Taylor expansion around λ = 0.

        The bias adjustment is calculated using a Taylor series expansion of the
        Box-Cox transformation around λ = 0.

        For the Box-Cox transformation:
        g(x, λ) = (x^λ - 1)/λ  for λ ≠ 0
        g(x, λ) = log(x)       for λ = 0

        The Taylor expansion around λ = 0 gives:
        g(x, λ) ≈ log(x) + λ/2 * (log(x))^2 + λ^2/6 * (log(x))^3 + O(λ^3)

        Parameters
        ----------
        y : pd.DataFrame
            Predictions in the original scale.
        variance : pd.DataFrame
            Variance of predictions in the transformed scale.

        Returns
        -------
        y_adjusted : pd.DataFrame
            Bias-adjusted predictions.
        """
        import numpy as np

        lmbda = self.boxcox_transformer_.lmbda_
        log_y = np.log(y)

        first_order = 0.5 * variance
        second_order = (lmbda * variance / 12) * (1 - 2 * log_y)
        third_order = (lmbda**2 * variance / 24) * (log_y**2 - 2 * log_y + 1)

        adjustment_factor = 1 + first_order + second_order + third_order
        y_adjusted = y * adjustment_factor

        return y_adjusted

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
