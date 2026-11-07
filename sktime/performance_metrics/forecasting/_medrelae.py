#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    median_relative_absolute_error,
)


class MedianRelativeAbsoluteError(BaseForecastingErrorMetricFunc):
    r"""Median relative absolute error (MdRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position.

    MdRAE applies median absolute error (MdAE) to the resulting relative errors.

    For a univariate, non-hierarchical sample of true values
    :math:`y_1, \dots, y_n`, predicted values
    :math:`\widehat{y}_1, \dots, \widehat{y}_n`, and benchmark predictions
    :math:`\widehat{y}_{b,1}, \dots, \widehat{y}_{b,n}` supplied as
    ``y_pred_benchmark``, define the absolute relative errors as

    .. math::

        r_i = \frac{|y_i - \widehat{y}_i|}
                   {\max(|y_i - \widehat{y}_{b,i}|, \epsilon)}.

    Here, :math:`\epsilon` is ``np.finfo(np.float64).eps``, used to avoid
    division by zero. A zero benchmark error can produce a large relative
    error when the forecast error is nonzero; if both errors are zero,
    :math:`r_i = 0`.

    With no time weights supplied, ``evaluate`` (or calling the metric
    with ``by_index=False``) returns

    .. math::

        \operatorname{MdRAE} = \operatorname{median}_{i=1,\dots,n} r_i.

    For an even number of time points, the median is the mean of the two
    central sorted values. If ``sample_weight`` (or ``horizon_weight``) is
    supplied to ``evaluate``, the lower weighted median is used instead.
    Lower values are better, and the minimum is zero.

    For multivariate data, the median is computed over time separately for
    each variable, then ``multioutput`` controls averaging across variables.
    ``multilevel`` controls aggregation across hierarchical series, see below.

    With no time weights supplied and :math:`n \geq 2`,
    ``evaluate_by_index`` uses the inherited jackknife calculation, returning
    at time index :math:`t_i` the pseudo-value

    .. math::

        p_i = n \operatorname{MdRAE}
              - (n - 1) \operatorname{MdRAE}_{-i},

    where :math:`\operatorname{MdRAE}_{-i}` is the metric evaluated after
    removing observation :math:`i` from the true values, predictions, and
    benchmark predictions. These pseudo-values are not the individual
    absolute relative errors :math:`r_i` and can be negative. The jackknife
    interpretation is not appropriate for median-based statistics.

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
        "capability:multivariate": True,
        "tests:skip_by_name": ["test_doctest_examples"],
    }

    func = median_relative_absolute_error
