#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics classes to assess performance on forecasting task.

Classes named as ``*Score`` return a value to maximize: the higher the better.
Classes named as ``*Error`` or ``*Loss`` return a value to minimize:
the lower the better.
"""

import numpy as np

from sktime.performance_metrics.forecasting._base import BaseForecastingErrorMetricFunc
from sktime.performance_metrics.forecasting._functions import (
    geometric_mean_absolute_error,
    geometric_mean_relative_absolute_error,
    geometric_mean_relative_squared_error,
    geometric_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_asymmetric_error,
    mean_linex_error,
    mean_relative_absolute_error,
    mean_squared_error,
    mean_squared_percentage_error,
    mean_squared_scaled_error,
    median_absolute_error,
    median_absolute_percentage_error,
    median_absolute_scaled_error,
    median_relative_absolute_error,
    median_squared_error,
    median_squared_percentage_error,
    median_squared_scaled_error,
    relative_loss,
)




class GeometricMeanRelativeSquaredError(BaseForecastingErrorMetricFunc):
    """Geometric mean relative squared error (GMRSE).

    If ``square_root`` is False then calculates GMRSE and if ``square_root`` is True
    then calculates root geometric mean relative squared error (RGMRSE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRSE applies geometric mean squared error (GMSE) to the resulting relative
    errors. RGMRSE applies root geometric mean squared error (RGMSE) to the
    resulting relative errors.

    Parameters
    ----------
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
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    GeometricMeanRelativeSquaredError
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrse = GeometricMeanRelativeSquaredError()
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.0008303544925949156)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.622419372049448)
    >>> gmrse = GeometricMeanRelativeSquaredError(multioutput='raw_values')
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([4.09227746e-06, 1.24483465e+00])
    >>> gmrse = GeometricMeanRelativeSquaredError(multioutput=[0.3, 0.7])
    >>> gmrse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.8713854839582426)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = geometric_mean_relative_squared_error

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        square_root=False,
        by_index=False,
    ):
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
        params2 = {"square_root": True}
        return [params1, params2]


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


class RelativeLoss(BaseForecastingErrorMetricFunc):
    """Calculate relative loss of forecast versus benchmark forecast.

    Applies a forecasting performance metric to a set of forecasts and
    benchmark forecasts and reports ratio of the metric from the forecasts to
    the the metric from the benchmark forecasts. Relative loss output is
    non-negative floating point. The best value is 0.0.

    If the score of the benchmark predictions for a given loss function is zero
    then a large value is returned.

    This function allows the calculation of scale-free relative loss metrics.
    Unlike mean absolute scaled error (MASE) the function calculates the
    scale-free metric relative to a defined loss function on a benchmark
    method instead of the in-sample training data. Like MASE, metrics created
    using this function can be used to compare forecast methods on a single
    series and also to compare forecast accuracy between series.

    This is useful when a scale-free comparison is beneficial but the training
    data used to generate some (or all) predictions is unknown such as when
    comparing the loss of 3rd party forecasts or surveys of professional
    forecasters.

    Only metrics that do not require y_train are currently supported.

    Parameters
    ----------
    relative_loss_function : function
        Function to use in calculation relative loss.

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

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import RelativeLoss
    >>> from sktime.performance_metrics.forecasting import mean_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_mae = RelativeLoss()
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.8148148148148147)
    >>> relative_mse = RelativeLoss(relative_loss_function=mean_squared_error)
    >>> relative_mse(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.5178095088655261)
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_mae = RelativeLoss()
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.8490566037735847)
    >>> relative_mae = RelativeLoss(multioutput='raw_values')
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    array([0.625     , 1.03448276])
    >>> relative_mae = RelativeLoss(multioutput=[0.3, 0.7])
    >>> relative_mae(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    np.float64(0.927272727272727)
    """

    _tags = {
        "requires-y-train": False,
        "requires-y-pred-benchmark": True,
        "univariate-only": False,
    }

    func = relative_loss

    def __init__(
        self,
        multioutput="uniform_average",
        multilevel="uniform_average",
        relative_loss_function=mean_absolute_error,
        by_index=False,
    ):
        self.relative_loss_function = relative_loss_function
        super().__init__(
            multioutput=multioutput,
            multilevel=multilevel,
            by_index=by_index,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Retrieve test parameters."""
        params1 = {}
        params2 = {"relative_loss_function": mean_squared_error}
        return [params1, params2]
