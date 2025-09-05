#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import numpy as np
import pandas as pd

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric


class MedianRelativeAbsoluteError(BaseForecastingErrorMetric):
    """Median relative absolute error (MdRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MdRAE applies medan absolute error (MdAE) to the resulting relative errors.

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
    MeanRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MedianRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mdrae = MedianRelativeAbsoluteError()
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(1.0)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.6944444444444443)
    >>> mdrae = MedianRelativeAbsoluteError(multioutput='raw_values')
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.55555556, 0.83333333])
    >>> mdrae = MedianRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> mdrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.7499999999999999)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    def _relative_absolute_error(self, y_true, y_pred, y_pred_benchmark, **kwargs):
        """Calculate the element-wise relative absolute error."""
        print("begin MedianRelativeAbsoluteError class")
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_pred_benchmark = np.asarray(y_pred_benchmark)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_pred_benchmark))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        y_pred_benchmark = y_pred_benchmark[mask]

        abs_bench_error = np.abs(y_true - y_pred_benchmark)
        eps = np.finfo(np.float64).eps
        denom = np.maximum(abs_bench_error, eps)
        relative_errors = np.abs(y_true - y_pred) / denom
        return relative_errors

    def _evaluate(self, y_true, y_pred, y_pred_benchmark, **kwargs):
        """Evaluate the median relative absolute error.

        Parameters
        ----------
        y_true : pandas.DataFrame or Series
            Ground truth (correct) target values.
        y_pred : pandas.DataFrame or Series
            Predicted values to evaluate.
        y_pred_benchmark : pandas.DataFrame or Series
            Benchmark values to evaluate.

        Returns
        -------
        loss : float or pd.Series
            Calculated metric, aggregated over time.
        """
        raw_values = self._relative_absolute_error(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_benchmark=y_pred_benchmark,
        )

        raw_values = self._get_weighted_df(raw_values, **kwargs)

        flat = (
            raw_values.values.flatten()
            if hasattr(raw_values, "values")
            else np.asarray(raw_values).flatten()
        )
        flat = flat[~np.isnan(flat)]

        if flat.size == 0:
            return np.nan
        return np.median(flat)

    def _evaluate_by_index(self, y_true, y_pred, y_pred_benchmark, **kwargs):
        """Return the metric evaluated at each time point.

        Parameters
        ----------
        y_true : pandas.DataFrame or Series
            Ground truth (correct) target values.
        y_pred : pandas.DataFrame or Series
            Predicted values to evaluate.
        y_pred_benchmark : pandas.DataFrame or Series
            Benchmark values to evaluate.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point.
        """
        multioutput = self.multioutput

        raw_values = self._relative_absolute_error(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_benchmark=y_pred_benchmark,
        )

        raw_values = self._get_weighted_df(raw_values, **kwargs)
        return self._handle_multioutput(raw_values, multioutput)

    def _get_weighted_df(self, raw_values, **kwargs):
        weights = kwargs.get("sample_weight", None)
        if weights is not None:
            if isinstance(raw_values, pd.DataFrame):
                raw_values = raw_values.mul(weights, axis=0)
            else:
                raw_values = raw_values * weights
        return raw_values
