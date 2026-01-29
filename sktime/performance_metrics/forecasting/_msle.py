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
    MSLE is sensitive to relative differences and is less sensitive to outliers
    in large values compared to standard MSE.

    The MSLE is defined as:

    .. math::
        \text{MSLE}(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n}
        (\log(1 + y_i) - \log(1 + \hat{y}_i))^2

    If ``square_root`` is True, the Root Mean Squared Logarithmic Error (RMSLE)
    is returned:

    .. math::
        \text{RMSLE}(y, \hat{y}) = \sqrt{\text{MSLE}(y, \hat{y})}

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like,
        default='uniform_average'
        Defines aggregating of multiple output values.
        If 'raw_values', returns errors for all outputs in multivariate case.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.
    multilevel : {'raw_values', 'uniform_average'}, default='uniform_average'
        Defines aggregating of multiple hierarchical levels.
        If 'raw_values', returns errors for all levels in hierarchical case.
        If 'uniform_average', errors are mean-averaged across levels.
    square_root : bool, default=False
        Whether to take the square root of the mean squared log error.
        If True, returns Root Mean Squared Log Error (RMSLE).
    by_index : bool, default=False
        If True, returns the metric value at each time point (jackknife pseudo-values).
        If False, returns the aggregate metric value.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import MeanSquaredLogError
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> msle = MeanSquaredLogError()
    >>> msle(y_true, y_pred)
    0.039730...
    """

    _tags = {
        "inner_implements_multilevel": True,  # Enables fast vectorized path
        "capability:multivariate": True,
    }

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

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Evaluate the desired metric on given inputs."""
        index_df = self._evaluate_by_index(y_true, y_pred, **kwargs)

        if self.multilevel == "raw_values":
            # Group by all levels except the last one (the time index)
            # This returns the mean error for each series in the hierarchy
            level_to_group = list(range(y_true.index.nlevels - 1))
            out_df = index_df.groupby(level=level_to_group).mean()
        else:
            # Default: Average across both time and hierarchy levels
            out_df = index_df.mean(axis=0)
        if self.square_root:
            out_df = np.sqrt(out_df)

        return out_df

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point."""
        y_true_np = np.asanyarray(y_true)
        y_pred_np = np.asanyarray(y_pred)

        # Numerical stability: MSLE is undefined for negative values
        y_true_np = np.maximum(y_true_np, 0)
        y_pred_np = np.maximum(y_pred_np, 0)

        y_true_log = np.log1p(y_true_np)
        y_pred_log = np.log1p(y_pred_np)

        squared_log_error = np.square(y_true_log - y_pred_log)

        # Reconstruct DataFrame with MultiIndex to preserve hierarchy
        squared_log_error = pd.DataFrame(
            squared_log_error, index=y_true.index, columns=y_true.columns
        )

        squared_log_error = self._get_weighted_df(squared_log_error, **kwargs)

        return self._handle_multioutput(squared_log_error, self.multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {"square_root": True, "multilevel": "raw_values"}
        return [params1, params2]
