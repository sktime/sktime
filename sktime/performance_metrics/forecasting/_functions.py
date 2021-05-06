# -*- coding: utf-8 -*-
"""Metrics to assess performance on forecasting task.
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better.
Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

# !/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
from scipy.stats import gmean
from sklearn.utils.stats import _weighted_percentile
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics._regression import _check_reg_targets
from sklearn.metrics import mean_absolute_error as _mean_absolute_error
from sklearn.metrics import mean_squared_error as _mean_squared_error
from sklearn.metrics import median_absolute_error as _median_absolute_error
from sktime.utils.validation.series import check_time_index, check_series

__author__ = ["Markus Löning", "Tomasz Chodakowski", "Ryan Kuhns"]
__all__ = [
    "relative_loss",
    "mean_asymmetric_error",
    "mean_absolute_scaled_error",
    "median_absolute_scaled_error",
    "mean_squared_scaled_error",
    "median_squared_scaled_error",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "median_squared_error",
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


def _weighted_geometric_mean(x, sample_weight=None, axis=None):
    """
    Parameters
    ----------
    array : np.ndarray
        Values to take the weighted geometric mean of.
    sample_weight: np.ndarray
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)`.

    Returns
    -------
    geometric_mean : float
        Weighted geometric mean
    """
    check_consistent_length(x, sample_weight)
    return np.exp(
        np.sum(sample_weight * np.log(x), axis=axis) / np.sum(sample_weight, axis=axis)
    )


def mean_asymmetric_error(
    y_true,
    y_pred,
    asymmetric_threshold=0.0,
    left_error_function="squared",
    right_error_function="absolute",
    horizon_weight=None,
    multioutput="uniform_average",
):
    """Calculates asymmetric loss function. Error values that are less
    than the asymmetric threshold have `left_error_function` applied.
    Error values greater than or equal to asymmetric threshold  have
    `right_error_function` applied.

    Many forecasting loss functions assume that over- and under-
    predictions should receive an equal penalty. However, this may not align
    with the actual cost faced by users' of the forecasts. Asymmetric loss
    functions are useful when the cost of under- and over- prediction are not
    the same.

    Setting `asymmetric_threshold` to zero, `left_error_function` to 'squared'
    and `right_error_function` to 'absolute` results in a greater penalty
    applied to over-predictions (y_true - y_pred < 0). The opposite is true
    for `left_error_function` set to 'absolute' and `right_error_function`
    set to 'squared`

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
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

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    asymmetric_loss : float
        Loss using asymmetric penalty of on errors.

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    ..[2]   Diebold, Francis X. (2007). "Elements of Forecasting (4th ed.)" ,
            Thomson, South-Western: Ohio, US.
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
    y_true, y_pred, y_train, sp=1, horizon_weight=None, multioutput="uniform_average"
):
    """Mean absolute scaled error (MASE). MASE output is non-negative floating
    point. The best value is 0.0.

    This scale-free error metric can be used to compare forecast methods on
    a single series and also to compare forecast accuracy between series.

    This metric is well suited to intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    y_true : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Estimated target values.

    y_train : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Observed training values.

    sp : int
        Seasonal periodicity of training data.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_absolute_scaled_error(y_true, y_pred, y_train)
    0.18333333333333335
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_absolute_scaled_error(y_true, y_pred, y_train)
    0.18181818181818182
    >>> mean_absolute_scaled_error(y_true, y_pred, y_train, multioutput='raw_values')
    array([0.10526316, 0.28571429])
    >>> mean_absolute_scaled_error(y_true, y_pred, y_train, multioutput=[0.3, 0.7])
    0.21935483870967742

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    ..[2]   Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
            for intermittent demand", Foresight, Issue 4.
    ..[3]   Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
            "The M4 Competition: 100,000 time series and 61 forecasting methods",
            International Journal of Forecasting, Volume 3
    """
    # Check if training set is prior to test set
    if isinstance(y_train, (pd.Series, pd.DataFrame)) and isinstance(
        y_true, (pd.Series, pd.DataFrame)
    ):
        check_time_index(y_train.index)
        if y_train.index.max() >= y_true.index.min():
            raise ValueError(
                "Found `y_train` with time index which is not "
                "before time index of `y_true`"
            )

    # Other input checks
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)
    y_train = check_series(y_train, enforce_univariate=False)
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
    y_true, y_pred, y_train, sp=1, horizon_weight=None, multioutput="uniform_average"
):
    """Median absolute scaled error (MdASE). MdASE output is non-negative
    floating point. The best value is 0.0.

    Taking the median instead of the mean of the test and train absolute errors
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Like MASE, this scale-free error metric can be used to compare forecast
    methods on a single series and also to compare forecast accuracy between
    series.

    Also like MASE, this metric is well suited to intermittent-demand series
    because it will not give infinite or undefined values unless the training
    data is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    y_true : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Estimated target values.

    y_train : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Observed training values.

    sp : int
        Seasonal periodicity of training data.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    See Also
    --------
    mean_absolute_scaled_error
    mean_squared_scaled_error
    median_squared_scaled_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import median_absolute_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> median_absolute_scaled_error(y_true, y_pred, y_train)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_absolute_scaled_error(y_true, y_pred, y_train)
    0.18181818181818182
    >>> median_absolute_scaled_error(y_true, y_pred, y_train, multioutput='raw_values')
    array([0.10526316, 0.28571429])
    >>> median_absolute_scaled_error(y_true, y_pred, y_train, multioutput=[0.3, 0.7])
    0.21935483870967742

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
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    ..[2]   Hyndman, R. J. (2006). "Another look at forecast accuracy metrics
            for intermittent demand", Foresight, Issue 4.
    ..[3]   Makridakis, S., Spiliotis, E. and Assimakopoulos, V. (2020)
            "The M4 Competition: 100,000 time series and 61 forecasting methods",
            International Journal of Forecasting, Volume 3
    """
    # Check if training set is prior to test set
    if isinstance(y_train, (pd.Series, pd.DataFrame)) and isinstance(
        y_true, (pd.Series, pd.DataFrame)
    ):
        check_time_index(y_train.index)
        if y_train.index.max() >= y_true.index.min():
            raise ValueError(
                "Found `y_train` with time index which is not "
                "before time index of `y_true`"
            )

    # Other input checks
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)
    y_train = check_series(y_train, enforce_univariate=False)
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
    y_train,
    sp=1,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
):
    """Mean squared scaled error (MSSE) `square_root` is False or
    root mean squared scaled error (RMSSE) if `square_root` is True.
    MSSE and RMSSE output is non-negative floating point. The best value is 0.0.

    This is a squared varient of the MASE loss metric. Like MASE this
    scale-free metric can be used to copmare forecast methods on a single
    series or between series.

    This metric is also suited for intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    y_true : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Estimated target values.

    y_train : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Observed training values.

    sp : int
        Seasonal periodicity of training data.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_squared_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_squared_scaled_error(y_true, y_pred, y_train, square_root=True)
    0.20568833780186058
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_squared_scaled_error(y_true, y_pred, y_train,  square_root=True)
    0.15679361328058636
    >>> mean_squared_scaled_error(y_true, y_pred, y_train, multioutput='raw_values',  \
    square_root=True)
    array([0.11215443, 0.20203051])
    >>> mean_squared_scaled_error(y_true, y_pred, y_train, multioutput=[0.3, 0.7], \
    square_root=True)
    0.17451891814894502

    References
    ----------
    ..[1]   M5 Competition Guidelines.
            https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx
    ..[2]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    # Check if training set is prior to test set
    if isinstance(y_train, (pd.Series, pd.DataFrame)) and isinstance(
        y_true, (pd.Series, pd.DataFrame)
    ):
        check_time_index(y_train.index)
        if y_train.index.max() >= y_true.index.min():
            raise ValueError(
                "Found `y_train` with time index which is not "
                "before time index of `y_true`"
            )
    # Other input checks
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)
    y_train = check_series(y_train, enforce_univariate=False)
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
    y_train,
    sp=1,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
):
    """Median squared scaled error (MdSSE) if `square_root` is False or
    root median squared scaled error (RMdSSE) if `square_root` is True.
    MdSSE and RMdSSE output is non-negative floating point. The best value is 0.0.

    This is a squared varient of the MdASE loss metric. Like MASE, MdASE, MSSE
    and RMSSE this scale-free metric can be used to compare forecast methods on a
    single series or between series.

    This metric is also suited for intermittent-demand series because it
    will not give infinite or undefined values unless the training data
    is a flat timeseries. In this case the function returns a large value
    instead of inf.

    Works with multioutput (multivariate) timeseries data
    with homogeneous seasonal periodicity.

    Parameters
    ----------
    y_true : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Estimated target values.

    y_train : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Observed training values.

    sp : int
        Seasonal periodicity of training data.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import median_squared_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_squared_scaled_error(y_true, y_pred, y_train, square_root=True)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_squared_scaled_error(y_true, y_pred, y_train, square_root=True)
    0.1472819539849714
    >>> median_squared_scaled_error(y_true, y_pred, y_train, multioutput='raw_values', \
    square_root=True)
    array([0.08687445, 0.20203051])
    >>> median_squared_scaled_error(y_true, y_pred, y_train, multioutput=[0.3, 0.7], \
    square_root=True)
    0.16914781383660782

    References
    ----------
    ..[1]   M5 Competition Guidelines.
            https://mofc.unic.ac.cy/wp-content/uploads/2020/03/M5-Competitors-Guide-Final-10-March-2020.docx
    ..[2]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    # Check if training set is prior to test set
    if isinstance(y_train, (pd.Series, pd.DataFrame)) and isinstance(
        y_true, (pd.Series, pd.DataFrame)
    ):
        check_time_index(y_train.index)
        if y_train.index.max() >= y_true.index.min():
            raise ValueError(
                "Found `y_train` with time index which is not "
                "before time index of `y_true`"
            )
    # Other input checks
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    if horizon_weight is not None:
        check_consistent_length(y_true, horizon_weight)
    y_train = check_series(y_train, enforce_univariate=False)
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
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Mean absolute error (MAE). MAE output is non-negative floating point.
    The best value is 0.0.

    MAE is on the same scale as the data. Because it takes the absolute value
    of the forecast error rather than the square, it is less sensitive to
    outliers than MSE or RMSE.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
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
):
    """Mean squared error (MSE) if `square_root` is False or
    root mean squared error (RMSE)  if `square_root` if True. MSE and RMSE are
    both non-negative floating point. The best value is 0.0.

    MSE is measured in squared units of the input data, and RMSE is on the
    same scale as the data. Because both metrics squares the
    forecast error rather than taking the absolute value, they are more sensitive
    to outliers than MAE or MdAE.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average' errors of all outputs are averaged with uniform weight.

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

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
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
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Median absolute error (MdAE).  MdAE output is non-negative floating
    point. The best value is 0.0.

    Like MAE, MdAE is on the same scale as the data. Because it takes the
    absolute value of the forecast error rather than the square, it is less
    sensitive to outliers than MSE, MdSE, RMSE or RMdSE.

    Taking the median instead of the mean of the absolute errors also makes
    this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
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
):
    """Median squared error (MdSE) if `square_root` is False or root median
    squared error (RMdSE) if `square_root` is True. MdSE and RMdSE return
    non-negative floating point. The best value is 0.0.

    Like MSE, MdSE is measured in squared units of the input data. RMdSe is
    on the same scale as the input data like RMSE. Because MdSE and RMdSE
    square the forecast error rather than taking the absolute value, they are
    more sensitive to outliers than MAE or MdAE.

    Taking the median instead of the mean of the squared errors makes
    this metric more robust to error outliers relative to a meean based metric
    since the median tends to be a more robust measure of central tendency in
    the presence of outliers.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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


    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
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


def mean_absolute_percentage_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average", symmetric=True
):
    """Mean absolute percentage error (MAPE) if `symmetric` is False or
    symmetric mean absolute percentage error (sMAPE) if `symmetric is True.
    MAPE and sMAPE output is non-negative floating point. The best value is 0.0.

    sMAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it is less sensitive to outliers than MSPE, RMSPE, MdSPE or RMdSPE.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    symmetric : bool, default=True
        Calculates symmetric version of metric if True.

    Returns
    -------
    loss : float
        MAPE or sMAPE loss.
        If multioutput is 'raw_values', then sMAPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average sMAPE of all output errors is returned.

    See Also
    --------
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
    mean_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_absolute_percentage_error(y_true, y_pred, symmetric=False)
    0.33690476190476193
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.5553379953379953
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_absolute_percentage_error(y_true, y_pred, symmetric=False)
    0.5515873015873016
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.6080808080808081
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values', \
        symmetric=False)
    array([0.38095238, 0.72222222])
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    array([0.71111111, 0.50505051])
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    symmetric=False)
    0.6198412698412699
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.5668686868686869

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
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
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average", symmetric=True
):
    """Median absolute percentage error (MdAPE) if `symmetric` is False or
    symmetric median absolute percentage error (sMdAPE). MdAPE and sMdAPE output
    is non-negative floating point. The best value is 0.0.

    MdAPE and sMdAPE are measured in percentage error relative to the test data.
    Because it takes the absolute value rather than square the percentage forecast
    error, it is less sensitive to outliers than MSPE, RMSPE, MdSPE or RMdSPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    symmetric : bool, default=True
        Calculates symmetric version of metric if True.

    Returns
    -------
    loss : float
        MdAPE or sMdAPE loss.
        If multioutput is 'raw_values', then MdAPE or sMdAPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdAPE or sMdAPE of all output errors is returned.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        median_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_absolute_percentage_error(y_true, y_pred, symmetric=False)
    0.16666666666666666
    >>> median_absolute_percentage_error(y_true, y_pred)
    0.18181818181818182
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_absolute_percentage_error(y_true, y_pred, symmetric=False)
    0.5714285714285714
    >>> median_absolute_percentage_error(y_true, y_pred)
    0.39999999999999997
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput='raw_values', \
    symmetric=False)
    array([0.14285714, 1.        ])
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    array([0.13333333, 0.66666667])
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7], \
    symmetric=False)
    0.7428571428571428
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.5066666666666666

    See Also
    --------
    mean_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
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
    symmetric=True,
):
    """Mean squared percentage error (MSPE) if `square_root` is False or
    root mean squared percentage error (RMSPE) if `square_root` is True.
    MSPE and RMSPE output is non-negative floating point. The best value is 0.0.

    MSPE is measured in squared percentage error relative to the test data and
    RMSPE is measured in percentage error relative to the test data.
    Because either calculation takes the square rather than absolute value of
    the percentage forecast error, they are more sensitive to outliers than
    MAPE, sMAPE, MdAPE or sMdAPE.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.s
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
    symmetric=True,
):
    """Median squared percentage error (MdSPE) if `square_root` is False or
    root median squared percentage error (RMdSPE) if `square_root` is True.
    MdSPE and RMdSPE output is non-negative floating point. The best value is 0.0.

    MdSPE is measured in squared percentage error relative to the test data.
    RMdSPE is measured in percentage error relative to the test data.
    Because it takes the square rather than absolute value of the percentage
    forecast error, both calculations are more sensitive to outliers than
    MAPE, sMAPE, MdAPE or sMdAPE.

    Taking the median instead of the mean of the absolute percentage errors also
    makes this metric more robust to error outliers since the median tends
    to be a more robust measure of central tendency in the presence of outliers.

    There is no limit on how large the error can be, particulalrly when `y_true`
    values are close to zero. In such cases the function returns a large value
    instead of `inf`.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.s
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
    y_true, y_pred, y_pred_benchmark, horizon_weight=None, multioutput="uniform_average"
):
    """Mean relative absolute error (MRAE).

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_relative_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mean_relative_absolute_error(y_true, y_pred, y_pred_benchmark)
    0.9511111111111111
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> mean_relative_absolute_error(y_true, y_pred, y_pred_benchmark)
    0.8703703703703702
    >>> mean_relative_absolute_error(y_true, y_pred, y_pred_benchmark, \
    multioutput='raw_values')
    array([0.51851852, 1.22222222])
    >>> mean_relative_absolute_error(y_true, y_pred, y_pred_benchmark, \
    multioutput=[0.3, 0.7])
    1.0111111111111108

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
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
    y_true, y_pred, y_pred_benchmark, horizon_weight=None, multioutput="uniform_average"
):
    """Median relative absolute error

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        median_relative_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> median_relative_absolute_error(y_true, y_pred, y_pred_benchmark)
    1.0
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> median_relative_absolute_error(y_true, y_pred, y_pred_benchmark)
    0.6944444444444443
    >>> median_relative_absolute_error(y_true, y_pred, y_pred_benchmark, \
    multioutput='raw_values')
    array([0.55555556, 0.83333333])
    >>> median_relative_absolute_error(y_true, y_pred, y_pred_benchmark, \
    multioutput=[0.3, 0.7])
    0.7499999999999999

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
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
    y_true, y_pred, y_pred_benchmark, horizon_weight=None, multioutput="uniform_average"
):
    """Geometric mean relative absolute error (GMRAE).

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        geometric_mean_relative_absolute_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> geometric_mean_relative_absolute_error(y_true, y_pred, y_pred_benchmark)
    0.0007839273064064755
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> geometric_mean_relative_absolute_error(y_true, y_pred, y_pred_benchmark)
    0.5578632807409556
    >>> geometric_mean_relative_absolute_error(y_true, y_pred, y_pred_benchmark, \
    multioutput='raw_values')
    array([4.97801163e-06, 1.11572158e+00])
    >>> geometric_mean_relative_absolute_error(y_true, y_pred, y_pred_benchmark, \
    multioutput=[0.3, 0.7])
    0.7810066018326863

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
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
            sample_weight=horizon_weight,
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
    y_pred_benchmark,
    horizon_weight=None,
    multioutput="uniform_average",
    square_root=False,
):
    """Geometric mean relative squared error (GMRSE) if `square_root` is False or
    root geometric mean relative squared error (RGMRSE) if `square_root` is True.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
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

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        geometric_mean_relative_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> y_pred_benchmark = y_pred*1.1
    >>> geometric_mean_relative_squared_error(y_true, y_pred, y_pred_benchmark)
    0.0008303544925949156
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> y_pred_benchmark = y_pred*1.1
    >>> geometric_mean_relative_squared_error(y_true, y_pred, y_pred_benchmark)
    0.622419372049448
    >>> geometric_mean_relative_squared_error(y_true, y_pred, y_pred_benchmark, \
    multioutput='raw_values')
    array([4.09227746e-06, 1.24483465e+00])
    >>> geometric_mean_relative_squared_error(y_true, y_pred, y_pred_benchmark, \
    multioutput=[0.3, 0.7])
    0.8713854839582426

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """

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
            sample_weight=horizon_weight,
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
    y_pred_benchmark,
    relative_loss_function=mean_absolute_error,
    horizon_weight=None,
    multioutput="uniform_average",
):
    """Calculates relative loss for a set of predictions and benchmark
    predictions for a given loss function. Relative loss output is non-negative
    floating point. The best value is 0.0.

    If the score of the benchmark predictions for a given loss function is zero
    then a large value is returned.

    This function allows the calculation of scale-free relative loss metrics.
    Unlike mean absolute scaled error (MASE) the function calculates the
    scale-free metric relative to a defined loss function on a benchmark
    method. Like MASE, metrics created using this function can be used to compare
    forecast methods on a single series and also to compare forecast accuracy
    between series.

    This is useful when a scale-free comparison is beneficial but the training
    used to generate some (or all) predictions is unknown such as when
    comparing the loss of 3rd party forecasts or surveys of professional
    forecastsers.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values from benchmark method.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        If 'raw_values', returns a full set of errors in case of multioutput input.
        If 'uniform_average', errors of all outputs are averaged with uniform weight.

    Returns
    -------
    relative_loss : float
        Loss for a method relative to loss for a benchmark method for a given
        loss metric.

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
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
):
    """Calculates asymmetric error.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    asymmetric_threshold : float, default = 0.0
        The value used to threshold the asymmetric loss function. Error values
        that are less than the asymmetric threshold have `left_error_function`
        applied. Error values greater than or equal to asymmetric threshold
        have `right_error_function` applied.

    left_error_function : str, {'squared', 'absolute'}
        Loss penalty to apply to error values less than the asymmetric threshold.

    right_error_function : str, {'squared', 'absolute'}
        Loss penalty to apply to error values greater than or equal to the
        asymmetric threshold.

    Returns
    -------
    asymmetric_errors : float
        Array of assymetric errors.

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """

    functions = {"squared": np.square, "absolute": np.abs}
    left_func, right_func = (
        functions[left_error_function],
        functions[right_error_function],
    )

    errors = np.where(
        y_true - y_pred < asymmetric_threshold,
        left_func(y_true - y_pred),
        right_func(y_true - y_pred),
    )
    return errors


def _relative_error(y_true, y_pred, y_pred_benchmark):
    """Relative error for each observations compared to  the error for that
    observation from a benchmark method.

    Parameters
    ----------
    y_true : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values.

    y_pred_benchmark : pandas Series, pandas DataFrame or NumPy array of
            shape (fh,) or (fh, n_outputs) where fh is the forecasting horizon
        Forecasted values from benchmark method.

    Returns
    -------
    relative_error : float
        relative error

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    denominator = np.where(
        y_true - y_pred_benchmark >= 0,
        np.maximum((y_true - y_pred_benchmark), EPS),
        np.minimum((y_true - y_pred_benchmark), -EPS),
    )
    return (y_true - y_pred) / denominator


def _percentage_error(y_true, y_pred, symmetric=True):
    """Percentage error.

    Parameters
    ----------
    y_true : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Ground truth (correct) target values.

    y_pred : pandas Series of shape (fh,) or (fh, n_outputs)
            where fh is the forecasting horizon
        Estimated target values.

    symmetric : bool, default = False
        Whether to calculate symmetric percentage error.

    Returns
    -------
    percentage_error : float

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
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
