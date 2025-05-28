#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import mean_asymmetric_error


class MeanAsymmetricError(BaseForecastingErrorMetricFunc):
    """Calculate mean of asymmetric loss function.

    Output is non-negative floating point. The best value is 0.0.

    Error values that are less than the asymmetric threshold have
    ``left_error_function`` applied. Error values greater than or equal to
    asymmetric threshold  have ``right_error_function`` applied.

    Many forecasting loss functions (like those discussed in [1]_) assume that
    over- and under- predictions should receive an equal penalty. However, this
    may not align with the actual cost faced by users' of the forecasts.
    Asymmetric loss functions are useful when the cost of under- and over-
    prediction are not the same.

    Setting ``asymmetric_threshold`` to zero, ``left_error_function`` to 'squared'
    and ``right_error_function`` to 'absolute` results in a greater penalty
    applied to over-predictions (y_true - y_pred < 0). The opposite is true
    for ``left_error_function`` set to 'absolute' and ``right_error_function``
    set to 'squared`.

    The left_error_penalty and right_error_penalty can be used to add differing
    multiplicative penalties to over-predictions and under-predictions.

    Parameters
    ----------
    asymmetric_threshold : float, default = 0.0
        The value used to threshold the asymmetric loss function. Error values
        that are less than the asymmetric threshold have ``left_error_function``
        applied. Error values greater than or equal to asymmetric threshold
        have ``right_error_function`` applied.
    left_error_function : {'squared', 'absolute'}, default='squared'
        Loss penalty to apply to error values less than the asymmetric threshold.
    right_error_function : {'squared', 'absolute'}, default='absolute'
        Loss penalty to apply to error values greater than or equal to the
        asymmetric threshold.
    left_error_penalty : int or float, default=1.0
        An additional multiplicative penalty to apply to error values less than
        the asymmetric threshold.
    right_error_penalty : int or float, default=1.0
        An additional multiplicative penalty to apply to error values greater
        than the asymmetric threshold.

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
    mean_linex_error

    Notes
    -----
    Setting ``left_error_function`` and ``right_error_function`` to "absolute", but
    choosing different values for ``left_error_penalty`` and ``right_error_penalty``
    results in the "lin-lin" error function discussed in [2]_.

    References
    ----------
    .. [1] Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
       forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    .. [2] Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
       Thomson, South-Western: Ohio, US.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import MeanAsymmetricError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> asymmetric_error = MeanAsymmetricError()
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.5)
    >>> asymmetric_error = MeanAsymmetricError(left_error_function='absolute', \
    right_error_function='squared')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.4625)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> asymmetric_error = MeanAsymmetricError()
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.75)
    >>> asymmetric_error = MeanAsymmetricError(left_error_function='absolute', \
    right_error_function='squared')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.7083333333333334)
    >>> asymmetric_error = MeanAsymmetricError(multioutput='raw_values')
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    array([0.5, 1. ])
    >>> asymmetric_error = MeanAsymmetricError(multioutput=[0.3, 0.7])
    >>> asymmetric_error(y_true, y_pred)  # doctest: +SKIP
    np.float64(0.85)
    """

    func = mean_asymmetric_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        asymmetric_threshold=0,
        left_error_function="squared",
        right_error_function="absolute",
        left_error_penalty=1.0,
        right_error_penalty=1.0,
        by_index=False,
    ):
        self.asymmetric_threshold = asymmetric_threshold
        self.left_error_function = left_error_function
        self.right_error_function = right_error_function
        self.left_error_penalty = left_error_penalty
        self.right_error_penalty = right_error_penalty

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
        params2 = {
            "asymmetric_threshold": 0.1,
            "left_error_function": "absolute",
            "right_error_function": "squared",
            "left_error_penalty": 2.0,
            "right_error_penalty": 0.5,
        }
        return [params1, params2]
