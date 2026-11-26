#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    geometric_mean_relative_absolute_error,
)


class GeometricMeanRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    r"""Geometric mean relative absolute error (GMRAE).

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n`,
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    and benchmark predictions
    :math:`\widehat{y}^{(b)}_1, \dots, \widehat{y}^{(b)}_n` (all in :math:`\mathbb{R}`),
    at time indices :math:`t_1, \dots, t_n`,
    the relative error at index :math:`t_i` is the ratio of the forecast error
    to the benchmark forecast error,
    :math:`r_i = \frac{y_i - \widehat{y}_i}{y_i - \widehat{y}^{(b)}_i}`.

    ``evaluate`` or call returns the Geometric Mean Relative Absolute Error,
    the geometric mean of the absolute relative errors,
    :math:`\left( \prod_{i=1}^n |r_i| \right)^{\frac{1}{n}}`.
    (the time indices are not used)

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`,
    jackknife pseudo-samples of the GMRAE at that time index,
    :math:`n \cdot \bar{\varepsilon} - (n-1) \cdot \varepsilon_i`,
    where :math:`\bar{\varepsilon}` is the GMRAE over all time indices,
    and :math:`\varepsilon_i` is the GMRAE with the i-th time index removed.

    GMRAE output is non-negative floating point. The best value is approximately
    zero, rather than zero.

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRAE applies geometric mean absolute error (GMAE) to the resulting relative
    errors.

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
    MedianRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

    Notes
    -----
    The geometric mean uses the product of values in its calculation. The presence
    of a zero value will result in the result being zero, even if all the other
    values are large. To partially account for this in the case where the forecast
    error is zero (an element of ``y_true`` equals the corresponding element of
    ``y_pred``), the resulting zero relative error values are replaced in the
    calculation with a small value. This results in the smallest value the metric
    can take being close to but not exactly zero.

    If the benchmark forecast error (``y_true`` minus ``y_pred_benchmark``) is
    zero, or close to zero, the denominator of the relative error is bounded away
    from zero by a small epsilon, so that a large relative error is returned at
    that time index rather than a division by zero.

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
        "capability:multivariate": True,
        "tests:skip_by_name": ["test_doctest_examples"],
    }

    func = geometric_mean_relative_absolute_error
