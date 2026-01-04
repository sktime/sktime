# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mean Squared Logarithmic Error (MSLE)."""

__author__ = ["alphaleporus"]

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
from sktime.performance_metrics.forecasting._functions import mean_squared_log_error


class MeanSquaredLogError(BaseForecastingErrorMetric):
    r"""Mean Squared Logarithmic Error (MSLE) or Root Mean Squared Log Error (RMSLE).

    MSLE is the Mean Squared Error calculated in logarithmic space.
    It is useful when targets have exponential growth (e.g., population, sales)
    or when we care more about relative errors than absolute errors.

    .. math::
        \text{MSLE} = \frac{1}{n} \sum_{i=1}^{n} (\log(1 + y_i) - \log(1 + \hat{y}_i))^2

    If ``square_root`` is True, calculates RMSLE:

    .. math::
        \text{RMSLE} = \sqrt{\text{MSLE}}

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple output values.
    multilevel : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple hierarchical levels.
    square_root : bool, default=False
        Whether to take the square root of the mean squared log error.
        If True, returns Root Mean Squared Log Error (RMSLE).
    by_index : bool, default=False
        If True, return the metric value at each time point.
        If False, return the aggregate metric value.
    """

    func = mean_squared_log_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point."""
        y_true_np = np.asanyarray(y_true)
        y_pred_np = np.asanyarray(y_pred)

        y_true_np = np.maximum(y_true_np, 0)
        y_pred_np = np.maximum(y_pred_np, 0)

        y_true_log = np.log1p(y_true_np)
        y_pred_log = np.log1p(y_pred_np)

        squared_log_error = np.square(y_true_log - y_pred_log)

        squared_log_error = pd.DataFrame(
            squared_log_error, index=y_true.index, columns=y_true.columns
        )

        squared_log_error = self._get_weighted_df(squared_log_error, **kwargs)

        return self._handle_multioutput(squared_log_error, self.multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"square_root": True}
        return [params1, params2]
