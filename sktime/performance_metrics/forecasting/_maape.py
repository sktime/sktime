# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mean Arctangent Absolute Percentage Error (MAAPE)."""

__author__ = ["alphaleporus"]

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
from sktime.performance_metrics.forecasting._common import _percentage_error


def mean_arctangent_absolute_percentage_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    relative_to="y_true",
    eps=None,
    **kwargs,
):
    """Mean Arctangent Absolute Percentage Error (MAAPE).

    MAAPE is a modification of MAPE that handles division by zero by applying arctan.
    It is bounded between 0 and pi/2.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs)
        Ground truth (correct) target values.
    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs)
        Estimated target values.
    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values.
    relative_to : {"y_true", "y_pred"}, default="y_true"
        Determines the denominator of the percentage error.
    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.

    Returns
    -------
    loss : float or ndarray of floats
        The computed metric value.
    """
    # Use _percentage_error from common utils to get the element-wise APE
    # This handles the epsilon and relative_to logic for us
    ape = _percentage_error(
        y_true=y_true,
        y_pred=y_pred,
        symmetric=False,  # MAAPE is defined on standard APE
        relative_to=relative_to,
        eps=eps,
    )

    # MAAPE Logic: arctan(APE)
    # arctan handles infinity gracefully (returns pi/2), solving the zero-division issue
    maape_values = np.arctan(ape)

    # Average across time (axis 0)
    output_errors = np.mean(maape_values, axis=0)

    if multioutput == "raw_values":
        return output_errors

    # FIX: np.average cannot take a string as weights.
    # If multioutput is "uniform_average", we set weights to None.
    weights = None if multioutput == "uniform_average" else multioutput

    return np.average(output_errors, weights=weights)


class MeanArctangentAbsolutePercentageError(BaseForecastingErrorMetric):
    """Mean Arctangent Absolute Percentage Error (MAAPE).

    MAAPE is a modification of MAPE that handles division by zero by applying arctan.
    It is bounded between 0 and pi/2.

    References
    ----------
    Kim, S., & Kim, H. (2016). "A new metric of absolute percentage error
    for intermittent demand forecasts". International Journal of Systems Science.
    """

    func = mean_arctangent_absolute_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        relative_to="y_true",
        eps=None,
    ):
        self.relative_to = relative_to
        self.eps = eps
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
        )

    def evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the metric."""
        return mean_arctangent_absolute_percentage_error(
            y_true=y_true,
            y_pred=y_pred,
            multioutput=self.multioutput,
            relative_to=self.relative_to,
            eps=self.eps,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"relative_to": "y_pred"}
        return [params1, params2]
