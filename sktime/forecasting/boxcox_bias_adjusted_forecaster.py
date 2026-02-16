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

    Notes
    -----
    This forecaster applies only to univariate, non-hierarchical inner forecasters.

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
        _DelegatedForecaster._set_delegated_tags(self, forecaster)
        super().__init__()

    def _get_delegate(self):
        """Return the delegate forecaster."""
        return self.forecaster

    def _fit(self, y, X=None, fh=None):
        """Fit the forecaster to the training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self.boxcox_transformer_ = BoxCoxTransformer(lambda_fixed=self.lambda_fixed)
        y_transformed = self.boxcox_transformer_.fit_transform(y)

        self.forecaster_ = self.forecaster.clone()
        self.forecaster_.fit(y=y_transformed, X=X, fh=fh)

        if not self.forecaster_.get_tag("capability:pred_int"):
            raise ValueError(
                "The wrapped forecaster must support prediction intervals "
                "(capability:pred_int) to enable bias adjustment."
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
        y_pred : pd.Series
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
        columns = pred_int_transformed.columns
        variance = self.forecaster_.predict_var(fh, X).squeeze()

        result = pd.DataFrame(0, index=pred_int_transformed.index, columns=columns)

        coverage_idx = 0
        var_type_idx = 1
        coverage_levels = columns.levels[coverage_idx].unique()

        for coverage_level in coverage_levels:
            level_mask = columns.get_level_values(coverage_idx) == coverage_level

            lower_mask = level_mask & (
                columns.get_level_values(var_type_idx) == "lower"
            )
            if any(lower_mask):
                lower = pred_int_transformed.loc[:, lower_mask]
                lower_adjusted = self._apply_bias_adjustment(lower, variance)
                result.loc[:, lower_mask] = lower_adjusted

            upper_mask = level_mask & (
                columns.get_level_values(var_type_idx) == "upper"
            )
            if any(upper_mask):
                upper = pred_int_transformed.loc[:, upper_mask]
                upper_adjusted = self._apply_bias_adjustment(upper, variance)
                result.loc[:, upper_mask] = upper_adjusted

        result = result.sort_index(axis=1)

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
        y_values = y.values.reshape(-1, 1)
        variance_values = variance.values.reshape(-1, 1)
        lmbda = self.boxcox_transformer_.lambda_

        denominator = 2 * (lmbda * y_values + 1) ** 2
        adjustment = 1 + (variance_values * (1 - lmbda)) / denominator
        adjusted_values = inv_boxcox(y_values, lmbda) * adjustment

        result = pd.Series(adjusted_values.ravel(), index=y.index, name=y.name)
        return result

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
