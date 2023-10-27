#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Metrics functions to assess performance on forecasting task.

Functions named as ``*_score`` return a value to maximize: the higher the better.
Function named as ``*_error`` or ``*_loss`` return a value to minimize:
the lower the better.
"""

import numpy as np
from scipy.stats import gmean
from sklearn.metrics import mean_absolute_error as _mean_absolute_error
from sklearn.metrics import mean_squared_error as _mean_squared_error
from sklearn.metrics import median_absolute_error as _median_absolute_error
from sklearn.metrics._regression import _check_reg_targets
from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.validation import check_consistent_length

from sktime.utils.stats import _weighted_geometric_mean

__author__ = ["mloning", "tch", "RNKuhns"]
__all__ = [
    "relative_loss",
    "mean_linex_error",
    "mean_asymmetric_error",
    "mean_absolute_scaled_error",
    "median_absolute_scaled_error",
    "mean_squared_scaled_error",
    "median_squared_scaled_error",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "median_squared_error",
    "geometric_mean_absolute_error",
    "geometric_mean_squared_error",
    "mean_absolute_percentage_error",
    "median_absolute_percentage_error",
    "mean_squared_percentage_error",
    "median_squared_percentage_error",
    "mean_relative_absolute_error",
    "median_relative_absolute_error",
    "geometric_mean_relative_absolute_error",
    "geometric_mean_relative_squared_error",
]

EPS = np.finfo(np.float64).eps


def _get_kwarg(kwarg, metric_name="Metric", **kwargs):
    """Pop a kwarg from kwargs and raise warning if kwarg not present."""
    kwarg_ = kwargs.pop(kwarg, None)
    if kwarg_ is None:
        msg = "".join(
            [
                f"{metric_name} requires `{kwarg}`. ",
                f"Pass `{kwarg}` as a keyword argument when calling the metric.",
            ]
        )
        raise ValueError(msg)
    return kwarg_


def mean_linex_error(
    y_true,
    y_pred,
    a=1.0,
    b=1.0,
    horizon_weight=None,
    multioutput="uniform_average",
    **kwargs,
):
    """Calculate mean linex error.

    Output is non-negative floating point. The best value is 0.0.

    Many forecasting loss functions (like those discussed in [1]_) assume that
    over- and under- predictions should receive an equal penalty. However, this
    may not align with the actual cost faced by users' of the forecasts.
    Asymmetric loss functions are useful when the cost of under- and over-
    prediction are not the same.

    The linex error function accounts for this by penalizing errors on one side
    of a threshold approximately linearly, while penalizing errors on the other
    side approximately exponentially.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.
    a : int or float
        Controls whether over- or under- predictions receive an approximately
        linear or exponential penalty. If `a` > 0 then negative errors
        (over-predictions) are penalized approximately linearly and positive errors
        (under-predictions) are penalized approximately exponentially. If `a` < 0
        the reverse is true.
    b : int or float
        Multiplicative penalty to apply to calculated errors.
    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    asymmetric_loss : float
        Loss using asymmetric penalty of on errors.
        If multioutput is 'raw_values', then asymmetric loss is returned for
        each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average asymmetric loss of all output errors is returned.

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
    >>> from sktime.performance_metrics.forecasting import mean_linex_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_linex_error(y_true, y_pred)  # doctest: +SKIP
    0.19802627763937575
    >>> mean_linex_error(y_true, y_pred, b=2)  # doctest: +SKIP
    0.3960525552787515
    >>> mean_linex_error(y_true, y_pred, a=-1)  # doctest: +SKIP
    0.2391800623225643
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_linex_error(y_true, y_pred)  # doctest: +SKIP
    0.2700398392309829
    >>> mean_linex_error(y_true, y_pred, a=-1)  # doctest: +SKIP
    0.49660966225813563
    >>> mean_linex_error(y_true, y_pred, multioutput='raw_values')  # doctest: +SKIP
    array([0.17220024, 0.36787944])
    >>> mean_linex_error(y_true, y_pred, multioutput=[0.3, 0.7])  # doctest: +SKIP
    0.30917568000716666
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    linex_error = _linex_error(y_true, y_pred, a=a, b=b)
    output_errors = np.average(linex_error, weights=horizon_weight, axis=0)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_asymmetric_error(
    y_true,
    y_pred,
    asymmetric_threshold=0.0,
    left_error_function="squared",
    right_error_function="absolute",
    left_error_penalty=1.0,
    right_error_penalty=1.0,
    horizon_weight=None,
    multioutput="uniform_average",
    **kwargs,
):
    """Calculate mean of asymmetric loss function.

    Output is non-negative floating point. The best value is 0.0.

    Error values that are less than the asymmetric threshold have
    `left_error_function` applied. Error values greater than or equal to
    asymmetric threshold  have `right_error_function` applied.

    Many forecasting loss functions (like those discussed in [1]_) assume that
    over- and under- predictions should receive an equal penalty. However, this
    may not align with the actual cost faced by users' of the forecasts.
    Asymmetric loss functions are useful when the cost of under- and over-
    prediction are not the same.

    Setting `asymmetric_threshold` to zero, `left_error_function` to 'squared'
    and `right_error_function` to 'absolute` results in a greater penalty
    applied to over-predictions (y_true - y_pred < 0). The opposite is true
    for `left_error_function` set to 'absolute' and `right_error_function`
    set to 'squared`.

    The left_error_penalty and right_error_penalty can be used to add differing
    multiplicative penalties to over-predictions and under-predictions.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.
    asymmetric_threshold : float, default = 0.0
        The value used to threshold the asymmetric loss function. Error values
        that are less than the asymmetric threshold have `left_error_function`
        applied. Error values greater than or equal to asymmetric threshold
        have `right_error_function` applied.
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
    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    asymmetric_loss : float
        Loss using asymmetric penalty of on errors.
        If multioutput is 'raw_values', then asymmetric loss is returned for
        each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average asymmetric loss of all output errors is returned.

    See Also
    --------
    mean_linex_error

    Notes
    -----
    Setting `left_error_function` and `right_error_function` to "absolute", but
    choosing different values for `left_error_penalty` and `right_error_penalty`
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
    >>> from sktime.performance_metrics.forecasting import mean_asymmetric_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_asymmetric_error(y_true, y_pred)
    0.5
    >>> mean_asymmetric_error(y_true, y_pred, left_error_function='absolute', \
    right_error_function='squared')
    0.4625
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_asymmetric_error(y_true, y_pred)
    0.75
    >>> mean_asymmetric_error(y_true, y_pred, left_error_function='absolute', \
    right_error_function='squared')
    0.7083333333333334
    >>> mean_asymmetric_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> mean_asymmetric_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)

    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    asymmetric_errors = _asymmetric_error(
        y_true,
        y_pred,
        asymmetric_threshold=asymmetric_threshold,
        left_error_function=left_error_function,
        right_error_function=right_error_function,
        left_error_penalty=left_error_penalty,
        right_error_penalty=right_error_penalty,
    )
    output_errors = np.average(asymmetric_errors, weights=horizon_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_absolute_scaled_error(
    y_true, y_pred, sp=1, horizon_weight=None, multioutput="uniform_average", **kwargs
):
    """Mean absolute scaled error (MASE).

    MASE output is non-negative floating point. The best value is 0.0.

    Like other scaled performance metrics, this scale-free error metric can be
    used to compare forecast methods on a single series and also to compare
    forecast accuracy between series.

    This metric is well suited to intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
             (n_timepoints, n_outputs), default = None
        Observed training values.

    sp : int
        Seasonal periodicity of training data.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.


    Returns
    -------
    loss : float or ndarray of floats
        MASE loss.
        If multioutput is 'raw_values', then MASE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MASE of all output errors is returned.

    See Also
    --------
    median_absolute_scaled_error
    mean_squared_scaled_error
    median_squared_scaled_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
    for intermittent demand", Foresight, Issue 4.

    Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
    "The M4 Competition: 100,000 time series and 61 forecasting methods",
    International Journal of Forecasting, Volume 3.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_absolute_scaled_error(y_true, y_pred, y_train=y_train)
    0.18333333333333335
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_absolute_scaled_error(y_true, y_pred, y_train=y_train)
    0.18181818181818182
    >>> mean_absolute_scaled_error(y_true, y_pred, y_train=y_train, \
    multioutput='raw_values')
    array([0.10526316, 0.28571429])
    >>> mean_absolute_scaled_error(y_true, y_pred, y_train=y_train, \
    multioutput=[0.3, 0.7])
    0.21935483870967742
    """
    y_train = _get_kwarg("y_train", metric_name="mean_absolute_scaled_error", **kwargs)

    # Other input checks
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    # _check_reg_targets converts 1-dim y_true,y_pred to 2-dim so need to match
    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, 1)

    # Check test and train have same dimensions
    if y_true.ndim != y_train.ndim:
        raise ValueError("Equal dimension required for y_true and y_train")

    if (y_true.ndim > 1) and (y_true.shape[1] != y_train.shape[1]):
        raise ValueError("Equal number of columns required for y_true and y_train")

    # naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]

    # mean absolute error of naive seasonal prediction
    mae_naive = mean_absolute_error(y_train[sp:], y_pred_naive, multioutput=multioutput)

    mae_pred = mean_absolute_error(
        y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
    )
    return mae_pred / np.maximum(mae_naive, EPS)


def median_absolute_scaled_error(
    y_true, y_pred, sp=1, horizon_weight=None, multioutput="uniform_average", **kwargs
):
    """Median absolute scaled error (MdASE).

    MdASE output is non-negative floating point. The best value is 0.0.

    Taking the median instead of the mean of the test and train absolute errors
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Like MASE and other scaled performance metrics this scale-free metric can be
    used to compare forecast methods on a single series or between series.

    Also like MASE, this metric is well suited to intermittent-demand series
    because it will not give infinite or undefined values unless the training
    data is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
             (n_timepoints, n_outputs), default = None
        Observed training values.

    sp : int
        Seasonal periodicity of training data.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    See Also
    --------
    mean_absolute_scaled_error
    mean_squared_scaled_error
    median_squared_scaled_error

    Returns
    -------
    loss : float or ndarray of floats
        MdASE loss.
        If multioutput is 'raw_values', then MdASE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdASE of all output errors is returned.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
    for intermittent demand", Foresight, Issue 4.

    Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
    "The M4 Competition: 100,000 time series and 61 forecasting methods",
    International Journal of Forecasting, Volume 3.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import median_absolute_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> median_absolute_scaled_error(y_true, y_pred, y_train=y_train)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_absolute_scaled_error(y_true, y_pred, y_train=y_train)
    0.18181818181818182
    >>> median_absolute_scaled_error(y_true, y_pred, y_train=y_train, \
    multioutput='raw_values')
    array([0.10526316, 0.28571429])
    >>> median_absolute_scaled_error( y_true, y_pred, y_train=y_train, \
    multioutput=[0.3, 0.7])
    0.21935483870967742
    """
    y_train = _get_kwarg(
        "y_train", metric_name="median_absolute_scaled_error", **kwargs
    )

    # Other input checks
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, 1)

    # Check test and train have same dimensions
    if y_true.ndim != y_train.ndim:
        raise ValueError("Equal dimension required for y_true and y_train")

    if (y_true.ndim > 1) and (y_true.shape[1] != y_train.shape[1]):
        raise ValueError("Equal number of columns required for y_true and y_train")

    # naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]

    # mean absolute error of naive seasonal prediction
    mdae_naive = median_absolute_error(
        y_train[sp:], y_pred_naive, multioutput=multioutput
    )

    mdae_pred = median_absolute_error(
        y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
    )
    return mdae_pred / np.maximum(mdae_naive, EPS)


def mean_squared_scaled_error(
    y_true,
    y_pred,
    sp=1,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
    **kwargs,
):
    """Mean squared scaled error (MSSE) or root mean squared scaled error (RMSSE).

    If `square_root` is False then calculates MSSE, otherwise calculates RMSSE if
    `square_root` is True. Both MSSE and RMSSE output is non-negative floating
    point. The best value is 0.0.

    This is a squared variant of the MASE loss metric.  Like MASE and other
    scaled performance metrics this scale-free metric can be used to compare
    forecast methods on a single series or between series.

    This metric is also suited for intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
             (n_timepoints, n_outputs), default = None
        Observed training values.

    sp : int
        Seasonal periodicity of training data.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    square_root : bool, default=False
        Whether to take the square root of the mean squared scaled error.
        If True, returns root mean squared scaled error (RMSSE)
        If False, returns mean squared scaled error (MSSE)

    Returns
    -------
    loss : float
        RMSSE loss.
        If multioutput is 'raw_values', then MSSE or RMSSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MSSE or RMSSE of all output errors is returned.

    See Also
    --------
    mean_absolute_scaled_error
    median_absolute_scaled_error
    median_squared_scaled_error

    References
    ----------
    M5 Competition Guidelines.
    https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx

    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_squared_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_squared_scaled_error(y_true, y_pred, y_train=y_train, square_root=True)
    0.20568833780186058
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_squared_scaled_error(y_true, y_pred, y_train=y_train,  square_root=True)
    0.15679361328058636
    >>> mean_squared_scaled_error(y_true, y_pred, y_train=y_train, \
    multioutput='raw_values', square_root=True)
    array([0.11215443, 0.20203051])
    >>> mean_squared_scaled_error(y_true, y_pred, y_train=y_train, \
    multioutput=[0.3, 0.7], square_root=True)
    0.17451891814894502
    """
    y_train = _get_kwarg("y_train", metric_name="mean_squared_scaled_error", **kwargs)

    # Other input checks
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, 1)

    # Check test and train have same dimensions
    if y_true.ndim != y_train.ndim:
        raise ValueError("Equal dimension required for y_true and y_train")

    if (y_true.ndim > 1) and (y_true.shape[1] != y_train.shape[1]):
        raise ValueError("Equal number of columns required for y_true and y_train")

    # naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]

    # mean squared error of naive seasonal prediction
    mse_naive = mean_squared_error(y_train[sp:], y_pred_naive, multioutput=multioutput)

    mse = mean_squared_error(
        y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
    )

    if square_root:
        loss = np.sqrt(mse / np.maximum(mse_naive, EPS))
    else:
        loss = mse / np.maximum(mse_naive, EPS)

    return loss


def median_squared_scaled_error(
    y_true,
    y_pred,
    sp=1,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
    **kwargs,
):
    """Median squared scaled error (MdSSE) or root median squared scaled error (RMdSSE).

    If `square_root` is False then calculates MdSSE, otherwise calculates RMdSSE if
    `square_root` is True. Both MdSSE and RMdSSE output is non-negative floating
    point. The best value is 0.0.

    This is a squared variant of the MdASE loss metric. Like MASE and other
    scaled performance metrics this scale-free metric can be used to compare
    forecast methods on a single series or between series.

    This metric is also suited for intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.
    y_train : pd.Series, pd.DataFrame or np.array of shape (n_timepoints,) or \
             (n_timepoints, n_outputs), default = None
        Observed training values.
    sp : int
        Seasonal periodicity of training data.
    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        RMdSSE loss.
        If multioutput is 'raw_values', then RMdSSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average RMdSSE of all output errors is returned.

    See Also
    --------
    mean_absolute_scaled_error
    median_absolute_scaled_error
    mean_squared_scaled_error

    References
    ----------
    M5 Competition Guidelines.
    https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx

    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import median_squared_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_squared_scaled_error(y_true, y_pred, y_train=y_train, square_root=True)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_squared_scaled_error(y_true, y_pred, y_train=y_train, square_root=True)
    0.1472819539849714
    >>> median_squared_scaled_error(y_true, y_pred, y_train=y_train, \
    multioutput='raw_values', square_root=True)
    array([0.08687445, 0.20203051])
    >>> median_squared_scaled_error(y_true, y_pred, y_train=y_train, \
    multioutput=[0.3, 0.7], square_root=True)
    0.16914781383660782
    """
    y_train = _get_kwarg("y_train", metric_name="median_squared_scaled_error", **kwargs)

    # Other input checks
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    if y_train.ndim == 1:
        y_train = np.expand_dims(y_train, 1)

    # Check test and train have same dimensions
    if y_true.ndim != y_train.ndim:
        raise ValueError("Equal dimension required for y_true and y_train")

    if (y_true.ndim > 1) and (y_true.shape[1] != y_train.shape[1]):
        raise ValueError("Equal number of columns required for y_true and y_train")

    # naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]

    # median squared error of naive seasonal prediction
    mdse_naive = median_squared_error(
        y_train[sp:], y_pred_naive, multioutput=multioutput
    )

    mdse = median_squared_error(
        y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
    )

    if square_root:
        loss = np.sqrt(mdse / np.maximum(mdse_naive, EPS))
    else:
        loss = mdse / np.maximum(mdse_naive, EPS)
    return loss


def mean_absolute_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average", **kwargs
):
    """Mean absolute error (MAE).

    MAE output is non-negative floating point. The best value is 0.0.

    MAE is on the same scale as the data. Because MAE takes the absolute value
    of the forecast error rather than squaring it, MAE penalizes large errors
    to a lesser degree than MSE or RMSE.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.
    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float or ndarray of floats
        MAE loss.
        If multioutput is 'raw_values', then MAE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MAE of all output errors is returned.

    See Also
    --------
    median_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_absolute_error
    geometric_mean_squared_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_absolute_error(y_true, y_pred)
    0.55
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_absolute_error(y_true, y_pred)
    0.75
    >>> mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85
    """
    return _mean_absolute_error(
        y_true, y_pred, sample_weight=horizon_weight, multioutput=multioutput
    )


def mean_squared_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
    **kwargs,
):
    """Mean squared error (MSE) or root mean squared error (RMSE).

    If `square_root` is False then calculates MSE and if `square_root` is True
    then RMSE is calculated.  Both MSE and RMSE are both non-negative floating
    point. The best value is 0.0.

    MSE is measured in squared units of the input data, and RMSE is on the
    same scale as the data. Because MSE and RMSE square the forecast error
    rather than taking the absolute value, they penalize large errors more than
    MAE.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root mean squared error (RMSE)
        If False, returns mean squared error (MSE)

    Returns
    -------
    loss : float or ndarray of floats
        MSE loss.
        If multioutput is 'raw_values', then MSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MSE of all output errors is returned.

    See Also
    --------
    mean_absolute_error
    median_absolute_error
    median_squared_error
    geometric_mean_absolute_error
    geometric_mean_squared_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_squared_error(y_true, y_pred)
    0.4125
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_squared_error(y_true, y_pred)
    0.7083333333333334
    >>> mean_squared_error(y_true, y_pred, square_root=True)
    0.8227486121839513
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.41666667, 1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values', square_root=True)
    array([0.64549722, 1.        ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.825
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7], square_root=True)
    0.8936491673103708
    """
    # Scikit-learn argument `squared` returns MSE when True and RMSE when False
    # Scikit-time argument `square_root` returns RMSE when True and MSE when False
    # Therefore need to pass the opposite of square_root as squared argument
    # to the scikit-learn function being wrapped
    squared = not square_root
    return _mean_squared_error(
        y_true,
        y_pred,
        sample_weight=horizon_weight,
        multioutput=multioutput,
        squared=squared,
    )


def median_absolute_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average", **kwargs
):
    """Median absolute error (MdAE).

    MdAE output is non-negative floating point. The best value is 0.0.

    Like MAE, MdAE is on the same scale as the data. Because MAE takes the
    absolute value of the forecast error rather than squaring it, MAE penalizes
    large errors to a lesser degree than MdSE or RdMSE.

    Taking the median instead of the mean of the absolute errors also makes
    this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        MdAE loss.
        If multioutput is 'raw_values', then MdAE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdAE of all output errors is returned.

    See Also
    --------
    mean_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_absolute_error
    geometric_mean_squared_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import median_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_absolute_error(y_true, y_pred)
    0.5
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_absolute_error(y_true, y_pred)
    0.75
    >>> median_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> median_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.85
    """
    return _median_absolute_error(
        y_true, y_pred, sample_weight=horizon_weight, multioutput=multioutput
    )


def median_squared_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
    **kwargs,
):
    """Median squared error (MdSE) or root median squared error (RMdSE).

    If `square_root` is False then calculates MdSE and if `square_root` is True
    then RMdSE is calculated. Both MdSE and RMdSE return non-negative floating
    point. The best value is 0.0.

    Like MSE, MdSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data like RMSE. Because MdSE and RMdSE
    square the forecast error rather than taking the absolute value, they
    penalize large errors more than MAE or MdAE.

    Taking the median instead of the mean of the squared errors makes
    this metric more robust to error outliers relative to a meean based metric
    since the median tends to be a more robust measure of central tendency in
    the presence of outliers.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root mean squared error (RMSE)
        If False, returns mean squared error (MSE)

    Returns
    -------
    loss : float
        MdSE loss.
        If multioutput is 'raw_values', then MdSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdSE of all output errors is returned.

    See Also
    --------
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    geometric_mean_absolute_error
    geometric_mean_squared_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import median_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_squared_error(y_true, y_pred)
    0.25
    >>> median_squared_error(y_true, y_pred, square_root=True)
    0.5
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_squared_error(y_true, y_pred)
    0.625
    >>> median_squared_error(y_true, y_pred, square_root=True)
    0.75
    >>> median_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.25, 1.  ])
    >>> median_squared_error(y_true, y_pred, multioutput='raw_values', square_root=True)
    array([0.5, 1. ])
    >>> median_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.7749999999999999
    >>> median_squared_error(y_true, y_pred, multioutput=[0.3, 0.7], square_root=True)
    0.85
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is None:
        output_errors = np.median(np.square(y_pred - y_true), axis=0)

    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = _weighted_percentile(
            np.square(y_pred - y_true), sample_weight=horizon_weight
        )

    if square_root:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def geometric_mean_absolute_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    **kwargs,
):
    """Geometric mean absolute error (GMAE).

    GMAE output is non-negative floating point. The best value is approximately
    zero, rather than zero.

    Like MAE and MdAE, GMAE is measured in the same units as the input data.
    Because GMAE takes the absolute value of the forecast error rather than
    squaring it, MAE penalizes large errors to a lesser degree than squared error
    variants like MSE, RMSE or GMSE or RGMSE.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        GMAE loss. If multioutput is 'raw_values', then GMAE is returned for each
        output separately. If multioutput is 'uniform_average' or an ndarray
        of weights, then the weighted average GMAE of all output errors is returned.

    See Also
    --------
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    geometric_mean_squared_error

    Notes
    -----
    The geometric mean uses the product of values in its calculation. The presence
    of a zero value will result in the result being zero, even if all the other
    values of large. To partially account for this in the case where elements
    of `y_true` and `y_pred` are equal (zero error), the resulting zero error
    values are replaced in the calculation with a small value. This results in
    the smallest value the metric can take (when `y_true` equals `y_pred`)
    being close to but not exactly zero.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    geometric_mean_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> geometric_mean_absolute_error(y_true, y_pred)
    0.000529527232030127
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> geometric_mean_absolute_error(y_true, y_pred)
    0.5000024031086919
    >>> geometric_mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    array([4.80621738e-06, 1.00000000e+00])
    >>> geometric_mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.7000014418652152
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    errors = y_true - y_pred
    errors = np.where(errors == 0.0, EPS, errors)
    if horizon_weight is None:
        output_errors = gmean(np.abs(errors), axis=0)
    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = _weighted_geometric_mean(
            np.abs(errors), weights=horizon_weight, axis=0
        )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def geometric_mean_squared_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
    **kwargs,
):
    """Geometric mean squared error (GMSE) or Root geometric mean squared error (RGMSE).

    If `square_root` is False then calculates GMSE and if `square_root` is True
    then RGMSE is calculated. Both GMSE and RGMSE return non-negative floating
    point. The best value is approximately zero, rather than zero.

    Like MSE and MdSE, GMSE is measured in squared units of the input data. RMdSE is
    on the same scale as the input data like RMSE and RdMSE. Because GMSE and RGMSE
    square the forecast error rather than taking the absolute value, they
    penalize large errors more than GMAE.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
                where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
                where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root geometric mean squared error (RGMSE)
        If False, returns geometric mean squared error (GMSE)

    Returns
    -------
    loss : float
        GMSE or RGMSE loss. If multioutput is 'raw_values', then loss is returned
        for each output separately. If multioutput is 'uniform_average' or an ndarray
        of weights, then the weighted average MdSE of all output errors is returned.

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
    of `y_true` and `y_pred` are equal (zero error), the resulting zero error
    values are replaced in the calculation with a small value. This results in
    the smallest value the metric can take (when `y_true` equals `y_pred`)
    being close to but not exactly zero.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import \
    geometric_mean_squared_error as gmse
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    2.80399089461488e-07
    >>> gmse(y_true, y_pred, square_root=True)  # doctest: +SKIP
    0.000529527232030127
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> gmse(y_true, y_pred)  # doctest: +SKIP
    0.5000000000115499
    >>> gmse(y_true, y_pred, square_root=True)  # doctest: +SKIP
    0.5000024031086919
    >>> gmse(y_true, y_pred, multioutput='raw_values')  # doctest: +SKIP
    array([2.30997255e-11, 1.00000000e+00])
    >>> gmse(y_true, y_pred, multioutput='raw_values', \
    square_root=True)  # doctest: +SKIP
    array([4.80621738e-06, 1.00000000e+00])
    >>> gmse(y_true, y_pred, multioutput=[0.3, 0.7])  # doctest: +SKIP
    0.7000000000069299
    >>> gmse(y_true, y_pred, multioutput=[0.3, 0.7], \
    square_root=True)  # doctest: +SKIP
    0.7000014418652152
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    errors = y_true - y_pred
    errors = np.where(errors == 0.0, EPS, errors)
    if horizon_weight is None:
        output_errors = gmean(np.square(errors), axis=0)
    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = _weighted_geometric_mean(
            np.square(errors), weights=horizon_weight, axis=0
        )

    if square_root:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_absolute_percentage_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    symmetric=False,
    **kwargs,
):
    """Mean absolute percentage error (MAPE) or symmetric version.

    If `symmetric` is False then calculates MAPE and if `symmetric` is True
    then calculates symmetric mean absolute percentage error (sMAPE). Both
    MAPE and sMAPE output is non-negative floating point. The best value is 0.0.

    sMAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    symmetric : bool, default=False
        Calculates symmetric version of metric if True.

    Returns
    -------
    loss : float
        MAPE or sMAPE loss.
        If multioutput is 'raw_values', then MAPE or sMAPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MAPE or sMAPE of all output errors is returned.

    See Also
    --------
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
    mean_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_absolute_percentage_error(y_true, y_pred, symmetric=False)
    0.33690476190476193
    >>> mean_absolute_percentage_error(y_true, y_pred, symmetric=True)
    0.5553379953379953
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_absolute_percentage_error(y_true, y_pred, symmetric=False)
    0.5515873015873016
    >>> mean_absolute_percentage_error(y_true, y_pred, symmetric=True)
    0.6080808080808081
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values', \
        symmetric=False)
    array([0.38095238, 0.72222222])
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values', \
        symmetric=True)
    array([0.71111111, 0.50505051])
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    symmetric=False)
    0.6198412698412699
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    symmetric=True)
    0.5668686868686869
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    output_errors = np.average(
        np.abs(_percentage_error(y_true, y_pred, symmetric=symmetric)),
        weights=horizon_weight,
        axis=0,
    )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def median_absolute_percentage_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    symmetric=False,
    **kwargs,
):
    """Median absolute percentage error (MdAPE) or symmetric version.

    If `symmetric` is False then calculates MdAPE and if `symmetric` is True
    then calculates symmetric median absolute percentage error (sMdAPE). Both
    MdAPE and sMdAPE output is non-negative floating point. The best value is 0.0.

    MdAPE and sMdAPE are measured in percentage error relative to the test data.
    Because it takes the absolute value rather than square the percentage forecast
    error, it penalizes large errors less than MSPE, RMSPE, MdSPE or RMdSPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    symmetric : bool, default=False
        Calculates symmetric version of metric if True.

    Returns
    -------
    loss : float
        MdAPE or sMdAPE loss.
        If multioutput is 'raw_values', then MdAPE or sMdAPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdAPE or sMdAPE of all output errors is returned.

    See Also
    --------
    mean_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        median_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_absolute_percentage_error(y_true, y_pred, symmetric=False)
    0.16666666666666666
    >>> median_absolute_percentage_error(y_true, y_pred, symmetric=True)
    0.18181818181818182
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_absolute_percentage_error(y_true, y_pred, symmetric=False)
    0.5714285714285714
    >>> median_absolute_percentage_error(y_true, y_pred, symmetric=True)
    0.39999999999999997
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput='raw_values', \
    symmetric=False)
    array([0.14285714, 1.        ])
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput='raw_values', \
    symmetric=True)
    array([0.13333333, 0.66666667])
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    symmetric=False)
    0.7428571428571428
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    symmetric=True)
    0.5066666666666666
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is None:
        output_errors = np.median(
            np.abs(_percentage_error(y_true, y_pred, symmetric=symmetric)), axis=0
        )
    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = _weighted_percentile(
            np.abs(_percentage_error(y_pred, y_true, symmetric=symmetric)),
            sample_weight=horizon_weight,
        )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_squared_percentage_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
    symmetric=False,
    **kwargs,
):
    """Mean squared percentage error (MSPE) or square root version.

    If `square_root` is False then calculates MSPE and if `square_root` is True
    then calculates root mean squared percentage error (RMSPE). If `symmetric`
    is True then calculates sMSPE or sRMSPE. Output is non-negative floating
    point. The best value is 0.0.

    MSPE is measured in squared percentage error relative to the test data and
    RMSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root mean squared error (RMSPE)
        If False, returns mean squared error (MSPE)

    symmetric : bool, default=False
        Calculates symmetric version of metric if True.

    Returns
    -------
    loss : float
        MSPE or RMSPE loss.
        If multioutput is 'raw_values', then MSPE or RMSPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MSPE or RMSPE of all output errors is returned.

    See Also
    --------
    mean_absolute_percentage_error
    median_absolute_percentage_error
    median_squared_percentage_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_squared_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_squared_percentage_error(y_true, y_pred, symmetric=False)
    0.23776218820861678
    >>> mean_squared_percentage_error(y_true, y_pred, square_root=True, \
    symmetric=False)
    0.48760864246710883
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_squared_percentage_error(y_true, y_pred, symmetric=False)
    0.5080309901738473
    >>> mean_squared_percentage_error(y_true, y_pred, square_root=True, \
    symmetric=False)
    0.7026794936195895
    >>> mean_squared_percentage_error(y_true, y_pred, multioutput='raw_values', \
    symmetric=False)
    array([0.34013605, 0.67592593])
    >>> mean_squared_percentage_error(y_true, y_pred, multioutput='raw_values', \
    square_root=True, symmetric=False)
    array([0.58321184, 0.82214714])
    >>> mean_squared_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    symmetric=False)
    0.5751889644746787
    >>> mean_squared_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    square_root=True, symmetric=False)
    0.7504665536595034
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    output_errors = np.average(
        np.square(_percentage_error(y_true, y_pred, symmetric=symmetric)),
        weights=horizon_weight,
        axis=0,
    )

    if square_root:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def median_squared_percentage_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
    symmetric=False,
    **kwargs,
):
    """Median squared percentage error (MdSPE)  or square root version.

    If `square_root` is False then calculates MdSPE and if `square_root` is True
    then calculates root median squared percentage error (RMdSPE). If `symmetric`
    is True then calculates sMdSPE or sRMdSPE. Output is non-negative floating
    point. The best value is 0.0.

    MdSPE is measured in squared percentage error relative to the test data.
    RMdSPE is measured in percentage error relative to the test data.
    Because the calculation takes the square rather than absolute value of
    the percentage forecast error, large errors are penalized more than
    MAPE, sMAPE, MdAPE or sMdAPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root mean squared error (RMSPE)
        If False, returns mean squared error (MSPE)

    symmetric : bool, default=False
        Calculates symmetric version of metric if True.

    Returns
    -------
    loss : float
        MdSPE or RMdSPE loss.
        If multioutput is 'raw_values', then MdSPE or RMdSPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdSPE or RMdSPE of all output errors is returned.

    See Also
    --------
    mean_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        median_squared_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_squared_percentage_error(y_true, y_pred, symmetric=False)
    0.027777777777777776
    >>> median_squared_percentage_error(y_true, y_pred, square_root=True, \
    symmetric=False)
    0.16666666666666666
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_squared_percentage_error(y_true, y_pred, symmetric=False)
    0.5102040816326531
    >>> median_squared_percentage_error(y_true, y_pred, square_root=True, \
    symmetric=False)
    0.5714285714285714
    >>> median_squared_percentage_error(y_true, y_pred, multioutput='raw_values', \
    symmetric=False)
    array([0.02040816, 1.        ])
    >>> median_squared_percentage_error(y_true, y_pred, multioutput='raw_values', \
    square_root=True, symmetric=False)
    array([0.14285714, 1.        ])
    >>> median_squared_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    symmetric=False)
    0.7061224489795918
    >>> median_squared_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    square_root=True, symmetric=False)
    0.7428571428571428
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    perc_err = _percentage_error(y_true, y_pred, symmetric=symmetric)
    if horizon_weight is None:
        output_errors = np.median(np.square(perc_err), axis=0)
    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = _weighted_percentile(
            np.square(perc_err),
            sample_weight=horizon_weight,
        )

    if square_root:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def mean_relative_absolute_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    **kwargs,
):
    """Mean relative absolute error (MRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MRAE applies mean absolute error (MAE) to the resulting relative errors.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
             (fh, n_outputs) where fh is the forecasting horizon, default=None
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        MRAE loss.
        If multioutput is 'raw_values', then MRAE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MRAE of all output errors is returned.

    See Also
    --------
    median_relative_absolute_error
    geometric_mean_relative_absolute_error
    geometric_mean_relative_squared_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_relative_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mean_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark)
    0.9511111111111111
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mean_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark)
    0.8703703703703702
    >>> mean_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark, multioutput='raw_values')
    array([0.51851852, 1.22222222])
    >>> mean_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark, multioutput=[0.3, 0.7])
    1.0111111111111108
    """
    y_pred_benchmark = _get_kwarg(
        "y_pred_benchmark", metric_name="mean_relative_absolute_error", **kwargs
    )
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    _, y_true, y_pred_benchmark, multioutput = _check_reg_targets(
        y_true, y_pred_benchmark, multioutput
    )

    if horizon_weight is None:
        output_errors = np.mean(
            np.abs(_relative_error(y_true, y_pred, y_pred_benchmark)), axis=0
        )
    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = np.average(
            np.abs(_relative_error(y_true, y_pred, y_pred_benchmark)),
            weights=horizon_weight,
            axis=0,
        )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def median_relative_absolute_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average", **kwargs
):
    """Median relative absolute error (MdRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    MdRAE applies medan absolute error (MdAE) to the resulting relative errors.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
             (fh, n_outputs) where fh is the forecasting horizon, default=None
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        MdRAE loss.
        If multioutput is 'raw_values', then MdRAE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdRAE of all output errors is returned.

    See Also
    --------
    mean_relative_absolute_error
    geometric_mean_relative_absolute_error
    geometric_mean_relative_squared_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        median_relative_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> median_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark)
    1.0
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> median_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark)
    0.6944444444444443
    >>> median_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark, multioutput='raw_values')
    array([0.55555556, 0.83333333])
    >>> median_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark, multioutput=[0.3, 0.7])
    0.7499999999999999
    """
    y_pred_benchmark = _get_kwarg(
        "y_pred_benchmark", metric_name="median_relative_absolute_error", **kwargs
    )
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    _, y_true, y_pred_benchmark, multioutput = _check_reg_targets(
        y_true, y_pred_benchmark, multioutput
    )

    if horizon_weight is None:
        output_errors = np.median(
            np.abs(_relative_error(y_true, y_pred, y_pred_benchmark)), axis=0
        )
    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = _weighted_percentile(
            np.abs(_relative_error(y_true, y_pred, y_pred_benchmark)),
            sample_weight=horizon_weight,
        )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def geometric_mean_relative_absolute_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    **kwargs,
):
    """Geometric mean relative absolute error (GMRAE).

    In relative error metrics, relative errors are first calculated by
    scaling (dividing) the individual forecast errors by the error calculated
    using a benchmark method at the same index position. If the error of the
    benchmark method is zero then a large value is returned.

    GMRAE applies geometric mean absolute error (GMAE) to the resulting relative
    errors.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
             (fh, n_outputs) where fh is the forecasting horizon, default=None
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        GMRAE loss.
        If multioutput is 'raw_values', then GMRAE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average GMRAE of all output errors is returned.

    See Also
    --------
    mean_relative_absolute_error
    median_relative_absolute_error
    geometric_mean_relative_squared_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        geometric_mean_relative_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> geometric_mean_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark)
    0.0007839273064064755
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> geometric_mean_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark)
    0.5578632807409556
    >>> geometric_mean_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark, multioutput='raw_values')
    array([4.97801163e-06, 1.11572158e+00])
    >>> geometric_mean_relative_absolute_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark, multioutput=[0.3, 0.7])
    0.7810066018326863
    """
    y_pred_benchmark = _get_kwarg(
        "y_pred_benchmark",
        metric_name="geometric_mean_relative_absolute_error",
        **kwargs,
    )
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    _, y_true, y_pred_benchmark, multioutput = _check_reg_targets(
        y_true, y_pred_benchmark, multioutput
    )

    relative_errors = np.abs(_relative_error(y_true, y_pred, y_pred_benchmark))
    if horizon_weight is None:
        output_errors = gmean(
            np.where(relative_errors == 0.0, EPS, relative_errors), axis=0
        )
    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = _weighted_geometric_mean(
            np.where(relative_errors == 0.0, EPS, relative_errors),
            weights=horizon_weight,
            axis=0,
        )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def geometric_mean_relative_squared_error(
    y_true,
    y_pred,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
    **kwargs,
):
    """Geometric mean relative squared error (GMRSE).

    If `square_root` is False then calculates GMRSE and if `square_root` is True
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
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
             (fh, n_outputs) where fh is the forecasting horizon, default=None
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    square_root : bool, default=False
        Whether to take the square root of the mean squared error.
        If True, returns root mean squared error (RMSPE)
        If False, returns mean squared error (MSPE)

    Returns
    -------
    loss : float
        GMRSE or RGMRSE loss.
        If multioutput is 'raw_values', then GMRSE or RGMRSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average GMRSE or RGMRSE of all output errors is returned.

    See Also
    --------
    mean_relative_absolute_error
    median_relative_absolute_error
    geometric_mean_relative_absolute_error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        geometric_mean_relative_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> geometric_mean_relative_squared_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark)
    0.0008303544925949156
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> geometric_mean_relative_squared_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark)
    0.622419372049448
    >>> geometric_mean_relative_squared_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark, multioutput='raw_values')
    array([4.09227746e-06, 1.24483465e+00])
    >>> geometric_mean_relative_squared_error(y_true, y_pred, \
    y_pred_benchmark=y_pred_benchmark, multioutput=[0.3, 0.7])
    0.8713854839582426
    """
    y_pred_benchmark = _get_kwarg(
        "y_pred_benchmark",
        metric_name="geometric_mean_relative_squared_error",
        **kwargs,
    )
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    _, y_true, y_pred_benchmark, multioutput = _check_reg_targets(
        y_true, y_pred_benchmark, multioutput
    )
    relative_errors = np.square(_relative_error(y_true, y_pred, y_pred_benchmark))
    if horizon_weight is None:
        output_errors = gmean(
            np.where(relative_errors == 0.0, EPS, relative_errors), axis=0
        )
    else:
        check_consistent_length(y_true, horizon_weight)
        output_errors = _weighted_geometric_mean(
            np.where(relative_errors == 0.0, EPS, relative_errors),
            weights=horizon_weight,
            axis=0,
        )

    if square_root:
        output_errors = np.sqrt(output_errors)

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def relative_loss(
    y_true,
    y_pred,
    relative_loss_function=mean_absolute_error,
    horizon_weight=None,
    multioutput="uniform_average",
    **kwargs,
):
    """Relative loss of forecast versus benchmark forecast for a given metric.

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
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
             (fh, n_outputs) where fh is the forecasting horizon, default=None
        Forecasted values from benchmark method.

    relative_loss_function : function, default=mean_absolute_error
        Function to use in calculation relative loss. The function must comply
        with API interface of sktime forecasting performance metrics. Metrics
        requiring y_train or y_pred_benchmark are not supported.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    relative_loss : float
        Loss for a method relative to loss for a benchmark method for a given
        loss metric.
        If multioutput is 'raw_values', then relative loss is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average relative loss of all output errors is returned.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.performance_metrics.forecasting import relative_loss
    >>> from sktime.performance_metrics.forecasting import mean_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_loss(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8148148148148147
    >>> relative_loss(y_true, y_pred, y_pred_benchmark=y_pred_benchmark, \
    relative_loss_function=mean_squared_error)
    0.5178095088655261
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> relative_loss(y_true, y_pred, y_pred_benchmark=y_pred_benchmark)
    0.8490566037735847
    >>> relative_loss(y_true, y_pred, y_pred_benchmark=y_pred_benchmark, \
    multioutput='raw_values')
    array([0.625     , 1.03448276])
    >>> relative_loss(y_true, y_pred, y_pred_benchmark=y_pred_benchmark, \
    multioutput=[0.3, 0.7])
    0.927272727272727
    """
    y_pred_benchmark = _get_kwarg(
        "y_pred_benchmark", metric_name="relative_loss", **kwargs
    )
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)

    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)

    loss_preds = relative_loss_function(
        y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
    )
    loss_benchmark = relative_loss_function(
        y_true,
        y_pred_benchmark,
        horizon_weight=horizon_weight,
        multioutput=multioutput,
    )
    return np.divide(loss_preds, np.maximum(loss_benchmark, EPS))


def _asymmetric_error(
    y_true,
    y_pred,
    asymmetric_threshold=0.0,
    left_error_function="squared",
    right_error_function="absolute",
    left_error_penalty=1.0,
    right_error_penalty=1.0,
):
    """Calculate asymmetric error.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.
    asymmetric_threshold : float, default = 0.0
        The value used to threshold the asymmetric loss function. Error values
        that are less than the asymmetric threshold have `left_error_function`
        applied. Error values greater than or equal to asymmetric threshold
        have `right_error_function` applied.
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

    Returns
    -------
    asymmetric_errors : float
        Array of asymmetric errors.

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.

    Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
    Thomson, South-Western: Ohio, US.
    """
    functions = {"squared": np.square, "absolute": np.abs}
    left_func, right_func = (
        functions[left_error_function],
        functions[right_error_function],
    )

    if not (
        isinstance(left_error_penalty, (int, float))
        and isinstance(right_error_penalty, (int, float))
    ):
        msg = "`left_error_penalty` and `right_error_penalty` must be int or float."
        raise ValueError(msg)

    errors = np.where(
        y_true - y_pred < asymmetric_threshold,
        left_error_penalty * left_func(y_true - y_pred),
        right_error_penalty * right_func(y_true - y_pred),
    )
    return errors


def _linex_error(y_true, y_pred, a=1.0, b=1.0):
    """Calculate mean linex error.

    Output is non-negative floating point. The best value is 0.0.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.
    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.
    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate metric for multivariate (multioutput) data.
        If array-like, values used as weights to average the errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    linex_error : float
        Array of linex errors.

    References
    ----------
    Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)",
    Thomson, South-Western: Ohio, US.
    """
    if not (isinstance(a, (int, float)) and a != 0):
        raise ValueError("`a` must be int or float not equal to zero.")
    if not (isinstance(b, (int, float)) and b > 0):
        raise ValueError("`b` must be an int or float greater than zero.")
    error = y_true - y_pred
    a_error = a * error
    linex_error = b * (np.exp(a_error) - a_error - 1)
    return linex_error


def _relative_error(y_true, y_pred, y_pred_benchmark):
    """Relative error for observations to benchmark method.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pd.Series, pd.DataFrame or np.array of shape (fh,) or \
             (fh, n_outputs) where fh is the forecasting horizon, default=None
        Forecasted values from benchmark method.

    Returns
    -------
    relative_error : float
        relative error

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of \
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """
    denominator = np.where(
        y_true - y_pred_benchmark >= 0,
        np.maximum((y_true - y_pred_benchmark), EPS),
        np.minimum((y_true - y_pred_benchmark), -EPS),
    )
    return (y_true - y_pred) / denominator


def _percentage_error(y_true, y_pred, symmetric=False):
    """Percentage error.

    Parameters
    ----------
    y_true : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pd.Series, pd.DataFrame or np.array of shape (fh,) or (fh, n_outputs) \
             where fh is the forecasting horizon
        Forecasted values.

    symmetric : bool, default = False
        Whether to calculate symmetric percentage error.

    Returns
    -------
    percentage_error : float

    References
    ----------
    Hyndman, R. J and Koehler, A. B. (2006). "Another look at measures of \
    forecast accuracy", International Journal of Forecasting, Volume 22, Issue 4.
    """
    if symmetric:
        # Alternatively could use np.abs(y_true + y_pred) in denom
        # Results will be different if y_true and y_pred have different signs
        percentage_error = (
            2
            * np.abs(y_true - y_pred)
            / np.maximum(np.abs(y_true) + np.abs(y_pred), EPS)
        )
    else:
        percentage_error = (y_true - y_pred) / np.maximum(np.abs(y_true), EPS)
    return percentage_error
