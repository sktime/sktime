#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    mean_squared_percentage_error,
)


class MeanSquaredPercentageError(BaseForecastingErrorMetricFunc):
    """Mean squared percentage error (MSPE), or RMSPE, or symmetric MSPE, RMSPE.

    If ``square_root`` is False then calculates MSPE and if ``square_root`` is True
    then calculates root mean squared percentage error (RMSPE). If ``symmetric``
    is True then calculates sMSPE or sRMSPE. Output is non-negative floating
    point. The best value is 0.0.

    MSPE is measured in squared percentage error relative to the test data and
    RMSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    There is no limit on how large the error can be, particularly when ``y_true``
    values are close to zero. In such cases the function returns a large value
    instead of ``inf``.

    Parameters
    ----------
    symmetric : bool, default = False
        Whether to calculate the symmetric version of the percentage metric
    square_root : bool, default = False
        Whether to take the square root of the metric

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
    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MedianSquaredPercentageError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    MeanSquaredPercentageError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mspe = MeanSquaredPercentageError(symmetric=False)
    >>> mspe(y_true, y_pred)
    np.float64(0.23776218820861678)
    >>> smspe = MeanSquaredPercentageError(square_root=True, symmetric=False)
    >>> smspe(y_true, y_pred)
    np.float64(0.48760864246710883)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mspe(y_true, y_pred)
    np.float64(0.5080309901738473)
    >>> smspe(y_true, y_pred)
    np.float64(0.7026794936195895)
    >>> mspe = MeanSquaredPercentageError(multioutput='raw_values', symmetric=False)
    >>> mspe(y_true, y_pred)
    array([0.34013605, 0.67592593])
    >>> smspe = MeanSquaredPercentageError(multioutput='raw_values', \
    symmetric=False, square_root=True)
    >>> smspe(y_true, y_pred)
    array([0.58321184, 0.82214714])
    >>> mspe = MeanSquaredPercentageError(multioutput=[0.3, 0.7], symmetric=False)
    >>> mspe(y_true, y_pred)
    np.float64(0.5751889644746787)
    >>> smspe = MeanSquaredPercentageError(multioutput=[0.3, 0.7], \
    symmetric=False, square_root=True)
    >>> smspe(y_true, y_pred)
    np.float64(0.7504665536595034)
    """

    func = mean_squared_percentage_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        symmetric=False,
        square_root=False,
        by_index=False,
    ):
        self.symmetric = symmetric
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

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
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {"symmetric": True, "square_root": True}
        return [params1, params2]
