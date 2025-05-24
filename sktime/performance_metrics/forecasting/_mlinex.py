#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import mean_linex_error


class MeanLinexError(BaseForecastingErrorMetricFunc):
    """Calculate mean linex error.

    Output is non-negative floating point. The best value is 0.0.

    Many forecasting loss functions (like those discussed in [1]_) assume that
    over- and under- predictions should receive an equal penalty. However, this
    may not align with the actual cost faced by users' of the forecasts.
    Asymmetric loss functions are useful when the cost of under- and over-
    prediction are not the same.

    The linex error function accounts for this by penalizing errors on one side
    of a threshold approximately linearly, while penalizing errors on the other
    side approximately exponentially. If ``a`` > 0 then negative errors
    (over-predictions) are penalized approximately linearly and positive errors
    (under-predictions) are penalized approximately exponentially. If ``a`` < 0
    the reverse is true.

    Parameters
    ----------
    a : int or float
        Controls whether over- or under- predictions receive an approximately
        linear or exponential penalty. If ``a`` > 0 then negative errors
        (over-predictions) are penalized approximately linearly and positive errors
        (under-predictions) are penalized approximately exponentially. If ``a`` < 0
        the reverse is true.
    b : int or float
        Multiplicative penalty to apply to calculated errors.

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
    mean_asymmetric_error

    Notes
    -----
    Calculated as b * (np.exp(a * error) - a * error - 1), where a != 0 and b > 0
    according to formula in [2]_.

    References
    ----------
    .. [1] Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
       forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    .. [1] Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
       Thomson, South-Western: Ohio, US.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanLinexError
    >>> linex_error = MeanLinexError()
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.19802627763937575)
    >>> linex_error = MeanLinexError(b=2)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.3960525552787515)
    >>> linex_error = MeanLinexError(a=-1)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.2391800623225643)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> linex_error = MeanLinexError()
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.2700398392309829)
    >>> linex_error = MeanLinexError(a=-1)
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.49660966225813563
    >>> linex_error = MeanLinexError(multioutput='raw_values')
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    array([0.17220024, 0.36787944])
    >>> linex_error = MeanLinexError(multioutput=[0.3, 0.7])
    >>> linex_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.30917568000716666)
    """

    func = mean_linex_error

    def __init__(
        self,
        a=1.0,
        b=1.0,
        multioutput="uniform_average",
        multilevel="uniform_average",
        by_index=False,
    ):
        self.a = a
        self.b = b
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
        params2 = {"a": 0.5, "b": 2}
        return [params1, params2]
