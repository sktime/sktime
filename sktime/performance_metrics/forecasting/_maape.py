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
    sample_weight=None,
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
    sample_weight : array-like of shape (fh,), default=None
        Synonym for horizon_weight, used for compatibility with sklearn interface.

    Returns
    -------
    loss : float or ndarray of floats
        The computed metric value.
    """
    # Use _percentage_error from common utils to get the element-wise APE
    ape = _percentage_error(
        y_true=y_true,
        y_pred=y_pred,
        symmetric=False,
        relative_to=relative_to,
        eps=eps,
    )

    # MAAPE Logic: arctan(abs(APE))
    maape_values = np.arctan(np.abs(ape))

    # Handle Weights: Prefer horizon_weight, fallback to sample_weight
    weights = horizon_weight if horizon_weight is not None else sample_weight

    # Average across time (axis 0)
    if weights is not None:
        output_errors = np.average(maape_values, axis=0, weights=weights)
    else:
        output_errors = np.mean(maape_values, axis=0)

    # Safe Check: Handle string vs array for multioutput
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return np.atleast_1d(output_errors)
        if multioutput == "uniform_average":
            return np.average(output_errors, weights=None)

    return np.average(output_errors, weights=multioutput)


class MeanArctangentAbsolutePercentageError(BaseForecastingErrorMetric):
    r"""Mean Arctangent Absolute Percentage Error (MAAPE).

    MAAPE is a variation of the Mean Absolute Percentage Error (MAPE) that is robust
    to zero values in the ground truth series. While MAPE is undefined when y_true=0,
    MAAPE uses the arctangent function to bound the error.

    The formula is defined as:

    .. math::
        \text{MAAPE} = \frac{1}{n} \sum_{t=1}^{n} \arctan \left(
        \left| \frac{y_t - \hat{y}_t}{y_t} \right| \right)

    where :math:`y_t` is the actual value and :math:`\hat{y}_t` is the forecast value.

    The result is bounded between 0 and :math:`\pi/2` (approx 1.57).

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values.
    multilevel : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple hierarchical levels.
    relative_to : {"y_true", "y_pred"}, default="y_true"
        Determines the denominator of the percentage error.
    eps : float, default=None
        Numerical epsilon used in denominator to avoid division by zero.
    by_index : bool, default=False
        If True, return the metric value at each time point.
        If False, return the aggregate metric value.

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
        by_index=False,
    ):
        self.relative_to = relative_to
        self.eps = eps
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point."""
        # 1. Calculate element-wise APE using standard sktime util
        ape = _percentage_error(
            y_true=y_true,
            y_pred=y_pred,
            symmetric=False,
            relative_to=self.relative_to,
            eps=self.eps,
        )

        # 2. Apply MAAPE logic: arctan(|APE|)
        maape_values = np.arctan(np.abs(ape))

        # 3. Handle weighting (Base class helper)
        maape_values = self._get_weighted_df(maape_values, **kwargs)

        # 4. Handle multioutput formatting (Base class helper)
        return self._handle_multioutput(maape_values, self.multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"relative_to": "y_pred", "multioutput": "raw_values"}
        return [params1, params2]
