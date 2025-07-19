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

    def _evaluate_by_index(self, y_true, y_pred, **kwargs):
        """Return the metric evaluated at each time point."""
        benchmark = self._get_benchmark(y_true, kwargs)
        y_true, y_pred, benchmark = self._align_indices(y_true, y_pred, benchmark)
        self._ensure_no_nan(benchmark)

        abs_error_pred = (y_true - y_pred).abs()
        abs_error_bench = (y_true - benchmark).abs()
        raw = self._compute_raw_values(abs_error_pred, abs_error_bench)
        weighted = self._get_weighted_df(raw, **kwargs)

        return self._handle_multioutput(weighted, self.multioutput)

    def _get_benchmark(self, y_true, kwargs):
        """Extract and validate the y_pred_benchmark argument."""
        bench = _get_kwarg(
            "y_pred_benchmark", metric_name="mean_relative_absolute_error", **kwargs
        )
        if (
            isinstance(bench, pd.DataFrame)
            and bench.shape[1] == 1
            and isinstance(y_true, pd.Series)
        ):
            bench = bench.iloc[:, 0]
        return bench

    def _align_indices(self, y_true, y_pred, benchmark):
        """
        Align indices or shapes between y_true, y_pred, and benchmark.

        Returns aligned y_true, y_pred, benchmark.
        """
        if hasattr(y_true, "index"):
            benchmark = self._align_to_index(y_true, benchmark)
            if hasattr(y_pred, "index"):
                y_pred = y_pred.reindex(benchmark.index)
        else:
            if hasattr(benchmark, "index"):
                benchmark = benchmark.values
        return y_true, y_pred, benchmark

    def _align_to_index(self, y_true, bench):
        idx = y_true.index
        if hasattr(bench, "index"):
            if bench.index.equals(idx):
                return bench
            overlap = idx.intersection(bench.index)
            if len(overlap) == 0:
                return self._align_by_position(y_true, bench)
            return bench.reindex(idx)
        else:
            arr = np.asarray(bench)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.flatten()
            if arr.shape[0] != len(idx):
                raise ValueError(
                    f"Benchmark length {arr.shape[0]} != y_true length {len(idx)}"
                )
            if isinstance(y_true, pd.Series):
                return pd.Series(arr, index=idx)
            return pd.DataFrame(arr, index=idx, columns=y_true.columns)

    def _align_by_position(self, y_true, bench):
        """Align when indices do not overlap: match by positional order."""
        idx = y_true.index
        arr = np.asarray(bench)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.flatten()
        if arr.shape[0] != len(idx):
            raise ValueError(
                f"Benchmark length {arr.shape[0]} != y_true length {len(idx)}"
            )
        if isinstance(y_true, pd.Series):
            return pd.Series(arr, index=idx)
        return pd.DataFrame(arr, index=idx, columns=y_true.columns)

    def _ensure_no_nan(self, bench):
        """Raise if benchmark contains NaN after alignment."""
        if hasattr(bench, "isna"):
            if bench.isna().any().any():
                raise ValueError(
                    "y_pred_benchmark contains NaN values after alignment."
                )
        else:
            if np.isnan(bench).any():
                raise ValueError("y_pred_benchmark contains NaN values.")

    def _compute_raw_values(self, abs_pred, abs_bench):
        """Compute raw relative absolute errors, handling zeros."""
        EPS = np.finfo(np.float64).eps
        both_zero = (abs_pred == 0) & (abs_bench == 0)
        safe_den = abs_bench.copy() if hasattr(abs_bench, "mask") else abs_bench
        safe_den = (
            safe_den.mask((abs_bench == 0) & ~both_zero, EPS)
            if hasattr(abs_bench, "mask")
            else np.where((abs_bench == 0) & ~both_zero, EPS, abs_bench)
        )
        raw = abs_pred / safe_den
        return (
            raw.mask(both_zero, 0)
            if hasattr(raw, "mask")
            else np.where(both_zero, 0, raw)
        )
