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


class MeanRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    r"""Mean relative absolute error (MRAE).

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n`,
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n`,
    and benchmark predictions
    :math:`\widehat{y}^{(b)}_1, \dots, \widehat{y}^{(b)}_n` (all in :math:`\mathbb{R}`),
    at time indices :math:`t_1, \dots, t_n`,
    the relative error at index :math:`t_i` is the ratio of the forecast error
    to the benchmark forecast error,
    :math:`r_i = \frac{y_i - \widehat{y}_i}{y_i - \widehat{y}^{(b)}_i}`.

    ``evaluate`` or call returns the Mean Relative Absolute Error,
    the arithmetic mean of the absolute relative errors,
    :math:`\frac{1}{n} \sum_{i=1}^n |r_i|`.
    (the time indices are not used)

    ``multioutput`` and ``multilevel`` control averaging across variables and
    hierarchy indices, see below.

    ``evaluate_by_index`` returns, at a time index :math:`t_i`,
    jackknife pseudo-samples of the MRAE at that time index,
    :math:`n \cdot \bar{\varepsilon} - (n-1) \cdot \varepsilon_i`,
    where :math:`\bar{\varepsilon}` is the MRAE over all time indices,
    and :math:`\varepsilon_i` is the MRAE with the i-th time index removed.
    As the MRAE is an arithmetic mean of pointwise terms, these pseudo-samples
    coincide with the pointwise absolute relative errors :math:`|r_i|`.

    MRAE output is non-negative floating point, with best possible value 0.0.

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

    Notes
    -----
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
        "tests:skip_by_name": ["test_doctest_examples"],
    }

    func = mean_relative_absolute_error
