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
from sktime.performance_metrics.forecasting._functions import (
    _get_kwarg,
    mean_relative_absolute_error,
)


class MeanRelativeAbsoluteError(BaseForecastingErrorMetric):
    """Mean relative absolute error (MRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MRAE applies mean absolute error (MAE) to the resulting relative errors.

    Parameters
    ----------
    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.

        * If array-like, values used as weights to average the errors.
        * If ``'raw_values'``,
          returns a full set of errors in case of multioutput input.
        * If ``'uniform_average'``,
          errors of all outputs are averaged with uniform weight.

    multilevel : {'raw_values', 'uniform_average', 'uniform_average_time'}
        Defines how to aggregate metric for hierarchical data (with levels).

        * If ``'uniform_average'`` (default),
          errors are mean-averaged across levels.
        * If ``'uniform_average_time'``,
          metric is applied to all data, ignoring level index.
        * If ``'raw_values'``,
          does not average errors across levels, hierarchy is retained.

    by_index : bool, default=False
        Determines averaging over time points in direct call to metric object.

        * If False, direct call to the metric object averages over time points,
          equivalent to a call of the``evaluate`` method.
        * If True, direct call to the metric object evaluates the metric at each
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
        "univariate-only": False,
    }

    func = mean_relative_absolute_error

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point.

        private _evaluate_by_index containing core logic, called from evaluate_by_index

        Parameters
        ----------
        y_true : time series in sktime compatible pandas based data container format
            Ground truth (correct) target values
            y can be in one of the following formats:
            Series scitype: pd.DataFrame
            Panel scitype: pd.DataFrame with 2-level row MultiIndex
            Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
        y_pred : time series in sktime compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed
        **kwargs : dict
            Additional keyword arguments, including y_pred_benchmark.

        Returns
        -------
        loss : pd.Series or pd.DataFrame
            Calculated metric, by time point (default=jackknife pseudo-values).
            pd.Series if self.multioutput="uniform_average" or array-like
            index is equal to index of y_true
            entry at index i is metric at time i, averaged over variables
            pd.DataFrame if self.multioutput="raw_values"
            index and columns equal to those of y_true
            i,j-th entry is metric at time i, at variable j
        """
        y_pred_benchmark = _get_kwarg(
            "y_pred_benchmark", metric_name="mean_relative_absolute_error", **kwargs
        )

        multioutput = self.multioutput

        if hasattr(y_true, "index") and hasattr(y_pred_benchmark, "index"):
            if hasattr(y_pred_benchmark, "ndim") and y_pred_benchmark.ndim == 2:
                if isinstance(y_true, pd.Series) and y_pred_benchmark.shape[1] == 1:
                    y_pred_benchmark = pd.Series(
                        y_pred_benchmark.iloc[:, 0], index=y_pred_benchmark.index
                    )
                elif (
                    isinstance(y_true, pd.DataFrame)
                    and y_pred_benchmark.shape[1] != y_true.shape[1]
                ):
                    raise ValueError(
                        f"y_pred_benchmark shape {y_pred_benchmark.shape} "
                        f"is incompatible with y_true shape {y_true.shape}"
                    )

            if not y_true.index.equals(y_pred_benchmark.index):
                overlapping_indices = y_true.index.intersection(y_pred_benchmark.index)
                if len(overlapping_indices) == 0:
                    if len(y_pred_benchmark) != len(y_true):
                        raise ValueError(
                            f"y_pred_benchmark length ({len(y_pred_benchmark)}) "
                            f"does not match y_true length ({len(y_true)}) "
                            f"and indices do not overlap"
                        )
                    if isinstance(y_true, pd.Series):
                        y_pred_benchmark = pd.Series(
                            y_pred_benchmark.values.flatten()
                            if hasattr(y_pred_benchmark.values, "flatten")
                            else y_pred_benchmark.values,
                            index=y_true.index,
                        )
                    else:
                        y_pred_benchmark = pd.DataFrame(
                            y_pred_benchmark.values,
                            index=y_true.index,
                            columns=y_true.columns,
                        )
                else:
                    y_pred_benchmark = y_pred_benchmark.reindex(y_true.index)

        elif hasattr(y_true, "index") and not hasattr(y_pred_benchmark, "index"):
            if hasattr(y_pred_benchmark, "ndim") and y_pred_benchmark.ndim == 2:
                if y_pred_benchmark.shape[1] == 1:
                    y_pred_benchmark = y_pred_benchmark.flatten()
                elif (
                    isinstance(y_true, pd.DataFrame)
                    and y_pred_benchmark.shape[1] == y_true.shape[1]
                ):
                    pass
                else:
                    raise ValueError(
                        f"y_pred_benchmark shape {y_pred_benchmark.shape} "
                        f"is incompatible with y_true shape {y_true.shape}"
                    )

            if len(y_pred_benchmark) != len(y_true):
                raise ValueError(
                    f"y_pred_benchmark length ({len(y_pred_benchmark)}) "
                    f"does not match y_true length ({len(y_true)})"
                )
            if isinstance(y_true, pd.Series):
                y_pred_benchmark = pd.Series(y_pred_benchmark, index=y_true.index)
            elif isinstance(y_true, pd.DataFrame):
                y_pred_benchmark = pd.DataFrame(
                    y_pred_benchmark, index=y_true.index, columns=y_true.columns
                )

        elif not hasattr(y_true, "index") and hasattr(y_pred_benchmark, "index"):
            y_pred_benchmark = y_pred_benchmark.values

        abs_error_pred = (y_true - y_pred).abs()
        abs_error_bench = (y_true - y_pred_benchmark).abs()
        EPS = np.finfo(np.float64).eps
        both_zero = (abs_error_pred == 0) & (abs_error_bench == 0)

        if hasattr(abs_error_bench, "isna"):
            if (
                abs_error_bench.isna().any().any()
                if hasattr(abs_error_bench.isna(), "any")
                else abs_error_bench.isna().any()
            ):
                raise ValueError(
                    "y_pred_benchmark contains NaN values after index alignment."
                )
        else:
            if np.isnan(abs_error_bench).any():
                raise ValueError("y_pred_benchmark contains NaN values.")

        if hasattr(abs_error_bench, "mask"):
            abs_error_bench_safe = abs_error_bench.copy()
            abs_error_bench_safe = abs_error_bench_safe.mask(
                (abs_error_bench == 0) & ~both_zero, EPS
            )
            raw_values = abs_error_pred / abs_error_bench_safe
            raw_values = raw_values.mask(both_zero, 0)
        else:
            abs_error_bench_safe = np.where(
                (abs_error_bench == 0) & ~both_zero, EPS, abs_error_bench
            )
            raw_values = abs_error_pred / abs_error_bench_safe
            raw_values = np.where(both_zero, 0, raw_values)

        raw_values = self._get_weighted_df(raw_values, **kwargs)
        return self._handle_multioutput(raw_values, multioutput)
