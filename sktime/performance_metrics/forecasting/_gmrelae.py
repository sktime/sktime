#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetric
from sktime.performance_metrics.forecasting._functions import (
    _get_kwarg,
)


class GeometricMeanRelativeAbsoluteError(BaseForecastingErrorMetric):
    """Geometric mean relative absolute error (GMRAE).

    This class implements an efficient, custom evaluation for GMRAE, including per-time-point
    evaluation via the `_evaluate_by_index` method. It does not rely on a function-based fallback.

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRAE applies geometric mean absolute error (GMAE) to the resulting relative
    errors.

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
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    GeometricMeanRelativeAbsoluteError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrae = GeometricMeanRelativeAbsoluteError()
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.0007839273064064755)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.5578632807409556)
    >>> gmrae = GeometricMeanRelativeAbsoluteError(multioutput='raw_values')
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([4.97801163e-06, 1.11572158e+00])
    >>> gmrae = GeometricMeanRelativeAbsoluteError(multioutput=[0.3, 0.7])
    >>> gmrae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.7810066018326863)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

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
        y_pred :time series in sktime compatible data container format
            Forecasted values to evaluate
            must be of same format as y_true, same indices and columns if indexed

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
        multioutput = self.multioutput

        y_pred_benchmark = _get_kwarg("y_pred_benchmark", **kwargs)
        
        # Calculate relative errors at each time point
        # For GMRAE, we need the absolute relative errors
        relative_errors = np.abs(self._relative_error(y_true, y_pred, y_pred_benchmark))
        
        EPS = np.finfo(np.float64).eps
        relative_errors = np.where(relative_errors == 0.0, EPS, relative_errors)
        
        relative_errors = self._get_weighted_df(relative_errors, **kwargs)
        
        return self._handle_multioutput(relative_errors, multioutput)
    
    def _relative_error(self, y_true, y_pred, y_pred_benchmark):
        """Calculate relative error for observations to benchmark method.
        
        Parameters
        ----------
        y_true : pd.DataFrame
            Ground truth (correct) target values
        y_pred : pd.DataFrame
            Forecasted values
        y_pred_benchmark : pd.DataFrame
            Forecasted values from benchmark method
            
        Returns
        -------
        relative_error : pd.DataFrame
            Relative errors
        """
        EPS = np.finfo(np.float64).eps
        
        denominator = np.where(
            y_true - y_pred_benchmark >= 0,
            np.maximum((y_true - y_pred_benchmark), EPS),
            np.minimum((y_true - y_pred_benchmark), -EPS),
        )
        
        return (y_true - y_pred) / denominator