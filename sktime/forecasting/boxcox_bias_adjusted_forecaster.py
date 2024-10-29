# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""BoxCoxBiasAdjustedForecaster implementation."""

__author__ = ["sanskarmodi8"]

import pandas as pd
from scipy.special import inv_boxcox

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.transformations.series.boxcox import BoxCoxTransformer


class BoxCoxBiasAdjustedForecaster(BaseForecaster):
    r"""Box-Cox Bias-Adjusted Forecaster.

    This module implements a forecaster that applies Box-Cox transformation
    and bias adjustment to the predictions of a wrapped forecaster.

    The bias adjustment is applied to both point predictions and prediction
    intervals using the bias correction formula for inverse Box-Cox transforms:

    .. math::

        y_t = \text{inv\_boxcox}(w_t, \lambda) \cdot [1 + \sigma_t^2(1-\lambda)/
        (2(\lambda w_t + 1)^2)]

    where:
    - :math:`y_t` is the back-transformed forecast at time t
    - :math:`w_t` is the transformed data at time t
    - :math:`\lambda` is the Box-Cox parameter
    - :math:`\sigma_t^2` is the forecast variance on the transformed scale at time t

    Parameters
    ----------
    forecaster : BaseForecaster
        The wrapped forecaster to which Box-Cox transformation and
        bias adjustment will be applied.
    lambda_fixed : float, optional (default=None)
        The Box-Cox transformation parameter. If None, it will be estimated.

    References
    ----------
    .. [1] Hyndman, R.J., & Athanasopoulos, G. (2018) "Forecasting:
           principles and practice", 2nd edition, Section 2.7 - Box-Cox Transformations,
           OTexts: Melbourne, Australia. OTexts.com/fpp2
    """

    _tags = {
        "capability:pred_int": True,
        "capability:pred_var": True,
    }

    def __init__(self, forecaster, lambda_fixed=None):
        self.forecaster = forecaster
        self.lambda_fixed = lambda_fixed
        self.boxcox_transformer_ = None
        self._y_name = None
        _DelegatedForecaster._set_delegated_tags(self, forecaster)
        super().__init__()

    def _get_delegate(self):
        """Return the delegate forecaster."""
        return self.forecaster

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to the training data.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self._y_name = y.name if isinstance(y, pd.Series) else None

        self.boxcox_transformer_ = BoxCoxTransformer(lambda_fixed=self.lambda_fixed)
        y_transformed = self.boxcox_transformer_.fit_transform(y)

        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=y_transformed, X=X, fh=fh)

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
        y_pred : pd.Series or pd.DataFrame
            Bias-adjusted point predictions.
        """
        y_pred_transformed = self.forecaster_.predict(fh, X)
        variance = self.forecaster_.predict_var(fh, X)

        y_pred = self._apply_bias_adjustment(y_pred_transformed, variance)
        if isinstance(y_pred, pd.DataFrame) and y_pred.shape[1] > 1:
            y_pred = y_pred.iloc[:, 0]

        if isinstance(y_pred, pd.Series):
            y_pred.name = self._y_name

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
        variance = self.forecaster_.predict_var(fh, X).squeeze()

        lower_adjusted = self._apply_bias_adjustment(
            pred_int_transformed.xs("lower", level=2, axis=1), variance
        )
        upper_adjusted = self._apply_bias_adjustment(
            pred_int_transformed.xs("upper", level=2, axis=1), variance
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
        original_index = y.index
        y = y.reset_index(drop=True)
        variance = variance.reset_index(drop=True)

        lmbda = self.boxcox_transformer_.lambda_

        denominator = 2 * (lmbda * y + 1) ** 2
        adjustment = 1 + (variance * (1 - lmbda)) / denominator

        adjusted_y = inv_boxcox(y, lmbda) * adjustment
        adjusted_y = pd.Series(adjusted_y.squeeze(), original_index)
        return adjusted_y

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

        params = [
            {
                "forecaster": NaiveForecaster(strategy="mean"),
            },
            {
                "forecaster": NaiveForecaster(strategy="mean"),
                "lambda_fixed": 0.5,
            },
        ]
        return params
