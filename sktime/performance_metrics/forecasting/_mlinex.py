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
    r"""Mean Linear Exponential (LinEx) error.

    Output is non-negative floating point. Smaller values are better,
    the minimal possible value is 0.0.

    The LinEx error is an asymmetric loss function, with parameter ``a``
    controlling the penalty for over- vs under-predictions.

    For a univariate, non-hierarchical sample
    of true values :math:`y_1, \dots, y_n` and
    predicted values :math:`\widehat{y}_1, \dots, \widehat{y}_n` (in :math:`mathbb{R}`),
    at time indices :math:`t_1, \dots, t_n`,
    ``evaluate`` or call returns the mean LinEx loss:

    .. math::

      \frac{b}{n}\sum_{i=1}^n \left( \exp(a \cdot e_i) - a \cdot e_i - 1 \right)

    where :math:`e_i = y_i - \widehat{y}_i`,
    and :math:`a \neq 0, b > 0` are parameters of the metric,
    ``a`` and ``b`` in the constructor.

    ``a`` controls the asymmetry of the penalty:

    - If ``a`` > 0, the penalty for over-predictions is approximately linear,
      while the penalty for under-predictions is approximately exponential.
    - If ``a`` < 0, the penalty for under-predictions is approximately linear,
      while the penalty for over-predictions is approximately exponential.

    ``b`` is a scale parameter that controls the overall magnitude of the penalty.

    ``multioutput`` and ``multilevel`` decide how results are averaged when
    there are multiple variables (multioutput) or hierarchical levels in the data.
    See below.

    ``evaluate_by_index`` returns, at a time index :math:`t_i` ,
    the LinEx loss at that time index,
    :math:`b \cdot (\exp(a \cdot e_i) - a \cdot e_i -1)` ,
    where :math:`e_i = y_i - \widehat{y}_i` ,
    for all time indices :math:`t_1, \dots, t_n` in the input.

    Parameters
    ----------
    a : int or float, default = 1
        Controls whether over- or under- predictions receive an approximately
        linear or exponential penalty. If ``a`` > 0 then negative errors
        (over-predictions) are penalized approximately linearly and positive errors
        (under-predictions) are penalized approximately exponentially. If ``a`` < 0
        the reverse is true.

    b : int or float, default = 1
        Multiplicative penalty to apply to calculated errors controlled
        by scale parameter.

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
    mean_asymmetric_error

    References
    ----------
    .. [1] Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
       forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    .. [2] Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
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
