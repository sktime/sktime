# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""BoxCoxBiasAdjustedForecaster implementation."""

__author__ = ["sanskarmodi8"]

import numpy as np
import pandas as pd
from scipy.special import inv_boxcox

from sktime.forecasting.base import BaseForecaster
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
    lmbda : float, optional (default=None)
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
        self.clone_tags(forecaster)
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
        y_pred : pd.DataFrame
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
    variance = self.forecaster_.predict_var(fh, X)

    if isinstance(pred_int_transformed.index, pd.PeriodIndex):
        pred_int_transformed.index = pred_int_transformed.index.to_timestamp()

    adjusted_intervals = {}

    for idx, cov, bound in pred_int_transformed.columns:
        adjusted_value = self._apply_bias_adjustment(
            pred_int_transformed[(idx, cov, bound)], variance
        )

        if isinstance(adjusted_value, np.ndarray):
            adjusted_value = adjusted_value.flatten()

        adjusted_intervals[(idx, cov, bound)] = adjusted_value

    result = pd.DataFrame(adjusted_intervals, index=pred_int_transformed.index)

    return result

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
        lmbda = self.boxcox_transformer_.lambda_

        denominator = 2 * (lmbda * y + 1) ** 2
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

        forecaster = NaiveForecaster()

        params1 = {"forecaster": forecaster}
        params2 = {"forecaster": forecaster, "lambda_fixed": 0.5}

        return [params1, params2]
