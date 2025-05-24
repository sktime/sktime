#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    geometric_mean_squared_error,
)


class GeometricMeanSquaredError(BaseForecastingErrorMetricFunc):
    """Geometric mean squared error (GMSE) or Root geometric mean squared error (RGMSE).

    If ``square_root`` is False then calculates GMSE and if ``square_root`` is True
    then RGMSE is calculated. Both GMSE and RGMSE return non-negative floating
    point. The best value is approximately zero, rather than zero.

    Like MSE and MdSE, GMSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data like RMSE and RdMSE. Because GMSE and RGMSE
    square the forecast error rather than taking the absolute value, they
    penalize large errors more than GMAE.

    Parameters
    ----------
    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root geometric mean squared error (RGMSE)
        If False, returns geometric mean squared error (GMSE)

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
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_absolute_error

    Notes
    -----
    The geometric mean uses the product of values in its calculation. The presence
    of a zero value will result in the result being zero, even if all the other
    values of large. To partially account for this in the case where elements
    of ``y_true`` and ``y_pred`` are equal (zero error), the resulting zero error
    values are replaced in the calculation with a small value. This results in
    the smallest value the metric can take (when ``y_true`` equals ``y_pred``)
    being close to but not exactly zero.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import GeometricMeanSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> gmse = GeometricMeanSquaredError()
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(2.80399089461488e-07)
    >>> rgmse = GeometricMeanSquaredError(square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.000529527232030127)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> gmse = GeometricMeanSquaredError()
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.5000000000115499)
    >>> rgmse = GeometricMeanSquaredError(square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.5000024031086919)
    >>> gmse = GeometricMeanSquaredError(multioutput='raw_values')
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    array([2.30997255e-11, 1.00000000e+00])
    >>> rgmse = GeometricMeanSquaredError(multioutput='raw_values', square_root=True)
    >>> rgmse(y_true, y_pred)# doctest: +SKIP
    array([4.80621738e-06, 1.00000000e+00])
    >>> gmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7])
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.7000000000069299)
    >>> rgmse = GeometricMeanSquaredError(multioutput=[0.3, 0.7], square_root=True)
    >>> rgmse(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.7000014418652152)
    """

    func = geometric_mean_squared_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
        self.square_root = square_root
        super().__init__(
            multioutput=multioutput, multilevel=multilevel, by_index=by_index
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
        params2 = {"square_root": True}
        return [params1, params2]
