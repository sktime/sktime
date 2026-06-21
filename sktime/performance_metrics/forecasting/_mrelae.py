#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    mean_relative_absolute_error,
)

import numpy as np
import pandas as pd 

class MeanRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    """Mean relative absolute error (MRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MRAE applies mean absolute error (MAE) to the resulting relative errors.

    Parameters
    ----------
    multioutput : 'uniform_average' (default), 1D array-like, or 'raw_values'
        Whether and how to aggregate metric for multivariate (multioutput) data.

        * If ``'uniform_average'`` (default),
          errors of all outputs are averaged with uniform weight.
        * If 1D array-like, errors are averaged across variables,
          with values used as averaging weights (same order).
        * If ``'raw_values'``,
          does not average across variables (outputs), per-variable errors are returned.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        How to aggregate the metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Controls averaging over time points in direct call to metric object.

        * If ``False`` (default),
          direct call to the metric object averages over time points,
          equivalent to a call of the ``evaluate`` method.
        * If ``True``, direct call to the metric object evaluates the metric at each
          time point, equivalent to a call of the ``evaluate_by_index`` method.

    See Also
    --------
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mrae = MeanRelativeAbsoluteError()
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.9511111111111111)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.8703703703703702)
    >>> mrae = MeanRelativeAbsoluteError(multioutput='raw_values')
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.51851852, 1.22222222])
    >>> mrae = MeanRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> mrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(1.0111111111111108)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "capability:multivariate": True,
    }

    func = mean_relative_absolute_error
 def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        Parameters
        ----------
        y_true : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Ground truth (correct) target values.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        y_pred : pandas.DataFrame with RangeIndex, integer index, or DatetimeIndex
            Predicted values to evaluate.
            Time series in sktime ``pd.DataFrame`` format for ``Series`` type.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point.

            * pd.Series if self.multioutput="uniform_average" or array-like;
              index is equal to index of y_true;
              entry at index i is metric at time i, averaged over variables.
            * pd.DataFrame if self.multioutput="raw_values";
              index and columns equal to those of y_true;
              i,j-th entry is metric at time i, at variable j.
        """
        multioutput = self.multioutput

        y_pred_benchmark = kwargs.pop("y_pred_benchmark")

        # eps to avoid division by zero, preserving sign of denominator
        eps = np.finfo(np.float64).eps
        denominator = np.where(
            y_true - y_pred_benchmark >= 0,
            np.maximum((y_true - y_pred_benchmark), eps),
            np.minimum((y_true - y_pred_benchmark), -eps),
        )

        # pointwise relative absolute error
        # mean is the aggregation across time points, dropped here
        raw_values = np.abs((y_true - y_pred) / denominator)

        # wrap back to DataFrame to preserve index/columns
        raw_values = pd.DataFrame(
            raw_values, index=y_true.index, columns=y_true.columns
        )

        raw_values = self._get_weighted_df(raw_values, **kwargs)

        return self._handle_multioutput(raw_values, multioutput)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params1 = {}
        params2 = {"by_index": True}
        return [params1, params2]
