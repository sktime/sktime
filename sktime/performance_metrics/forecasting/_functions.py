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

# from sklearn.metrics import (
# mean_absolute_error, mean_squared_error, median_absolute_error,
# mean_absolute_percentage_error
# )
from sktime.utils.validation.series import check_time_index
from sktime.utils.validation.forecasting import (
    check_y,
    check_y_true_pred,
    check_horizon_weights,
)

__author__ = ["Markus Löning", "Tomasz Chodakowski", "Ryan Kuhns"]
__all__ = [
    "relative_loss",
    "mean_asymmetric_error",
    "mean_absolute_scaled_error",
    "median_absolute_scaled_error",
    "root_mean_squared_scaled_error",
    "root_median_squared_scaled_error",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "median_absolute_error",
    "median_squared_error",
    "root_median_squared_error",
    "symmetric_mean_absolute_percentage_error",
    "symmetric_median_absolute_percentage_error",
    "mean_absolute_percentage_error",
    "median_absolute_percentage_error",
    "mean_squared_percentage_error",
    "median_squared_percentage_error",
    "root_mean_squared_percentage_error",
    "root_median_squared_percentage_error",
    "mean_relative_absolute_error",
    "median_relative_absolute_error",
    "geometric_mean_relative_absolute_error",
    "geometric_mean_relative_squared_error",
]

EPS = np.finfo(np.float64).eps


def weighted_geometric_mean(x, sample_weights=None, axis=None):
    """
    Parameters
    ----------
    array : 1D or 2D array
        Values to take the weighted geometric mean of.
    sample_weight: 1D or 2D array
        Weights for each value in `array`. Must be same shape as `array` or
        of shape `(array.shape[0],)`.

    Returns
    -------
    geometric_mean : float
        Weighted geometric mean
    """
    # TODO: ADD Checks of inputs
    return np.exp(
        np.sum(sample_weights * np.log(x), axis=axis)
        / np.sum(sample_weights, axis=axis)
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

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is not None:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)

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
    y_true, y_pred, y_train, sp=1, multioutput="uniform_average"
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

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    root_mean_squared_scaled_error
    root_median_squared_scaled_error

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
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    y_train = check_y(y_train, enforce_univariate=False, allow_numpy=True)

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

    mae_pred = mean_absolute_error(y_true, y_pred, multioutput=multioutput)
    return mae_pred / np.maximum(mae_naive, EPS)


def median_absolute_scaled_error(
    y_true, y_pred, y_train, sp=1, multioutput="uniform_average"
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

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    See Also
    --------
    mean_absolute_scaled_error
    root_mean_squared_scaled_error
    root_median_squared_scaled_error

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
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    y_train = check_y(y_train, enforce_univariate=False, allow_numpy=True)

    # Check test and train have same dimensions
    if y_true.ndim != y_train.ndim:
        raise ValueError("Equal dimension required for y_true and y_train")

    if (y_true.ndim > 1) and (y_true.shape[1] != y_train.shape[1]):
        raise ValueError("Equal number of columns required for y_true and y_train")

    #  naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]

    # mean absolute error of naive seasonal prediction
    mdae_naive = median_absolute_error(
        y_train[sp:], y_pred_naive, multioutput=multioutput
    )

    mdae_pred = median_absolute_error(y_true, y_pred, multioutput=multioutput)
    return mdae_pred / np.maximum(mdae_naive, EPS)


def root_mean_squared_scaled_error(
    y_true, y_pred, y_train, sp=1, horizon_weight=None, multioutput="uniform_average"
):
    """Root mean squared scaled error (RMSSE). RMSSE output is non-negative
        floating point. The best value is 0.0.

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

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        RMSSE loss.
        If multioutput is 'raw_values', then RMSSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average RMSSE of all output errors is returned.

    See Also
    --------
    mean_absolute_scaled_error
    median_absolute_scaled_error
    root_median_squared_scaled_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        root_mean_squared_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> root_mean_squared_scaled_error(y_true, y_pred, y_train)
    0.20568833780186058
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> root_mean_squared_scaled_error(y_true, y_pred, y_train)
    0.15679361328058636
    >>> root_mean_squared_scaled_error(
            y_true, y_pred, y_train, multioutput='raw_values'
        )
    array([0.11215443, 0.20203051])
    >>> root_mean_squared_scaled_error(
            y_true, y_pred, y_train, multioutput=[0.3, 0.7]
        )
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
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    y_train = check_y(y_train, enforce_univariate=False, allow_numpy=True)

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

    mse = mean_squared_error(y_true, y_pred, multioutput=multioutput)

    return np.sqrt(mse / np.maximum(mse_naive, EPS))


def root_median_squared_scaled_error(
    y_true, y_pred, y_train, sp=1, horizon_weight=None, multioutput="uniform_average"
):
    """Root median squared scaled error (RMdSSE). RMdSSE output is non-negative
        floating point. The best value is 0.0.

    This is a squared varient of the MdASE loss metric. Like MASE, MdASE and
    RMSSE this scale-free metric can be used to copmare forecast methods on a
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

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    root_mean_squared_scaled_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        root_median_squared_scaled_error
    >>> y_train = np.array([5, 0.5, 4, 6, 3, 5, 2])
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> root_median_squared_scaled_error(y_true, y_pred, y_train)
    0.16666666666666666
    >>> y_train = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> root_median_squared_scaled_error(y_true, y_pred, y_train)
    0.1472819539849714
    >>> root_median_squared_scaled_error(
            y_true, y_pred, y_train, multioutput='raw_values'
        )
    array([0.08687445, 0.20203051])
    >>> root_median_squared_scaled_error(
            y_true, y_pred, y_train, multioutput=[0.3, 0.7]
        )
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
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    y_train = check_y(y_train, enforce_univariate=False, allow_numpy=True)

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

    mdse = median_squared_error(y_true, y_pred, multioutput=multioutput)

    return np.sqrt(mdse / np.maximum(mdse_naive, EPS))


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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    root_mean_squared_error
    root_median_squared_error

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

    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is not None:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
    # Once Scikit-learn 0.24 is widely available through Conda then switch
    # to importing Scikit's function, which I used the code from below in the
    # interim
    # return mean_absolute_error(
    # y_true, y_pred, sample_weight=horizon_weight,multioutput=multioutput
    # )
    output_errors = np.average(np.abs(y_true - y_pred), weights=horizon_weight, axis=0)
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def root_mean_squared_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Root mean squared error (RMSE). RMSE output is non-negative floating
    point. The best value is 0.0.

    RMSE is on same scale as the data. Because it squares the
    forecast error rather than taking the absolute value, it is more sensitive
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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        RMSE loss.
        If multioutput is 'raw_values', then RMSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average RMSE of all output errors is returned.

    See Also
    --------
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    root_median_squared_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import root_mean_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> root_mean_squared_error(y_true, y_pred)
    0.6422616289332564
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> root_mean_squared_error(y_true, y_pred)
    0.6422616289332564
    >>> root_mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.64549722, 1. ])
    >>> root_mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.9082951062292475

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    mse = mean_squared_error(
        y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
    )
    return np.sqrt(mse)


def mean_squared_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Mean squared error (MSE). MSE output is non-negative floating point.
    The best value is 0.0.

    MSE is measured in squared units of the input data. Because it squares the
    forecast error rather than taking the absolute value, it is more sensitive
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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.


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
    root_mean_squared_error
    root_median_squared_error

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
    >>> mean_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.41666667, 1. ])
    >>> mean_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.825

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is not None:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
    # Once Scikit-learn 0.24 is widely available through Conda then switch
    # to importing Scikit's function, which I used the code from below in the
    # interim
    # return mean_squared_error(
    # y_true, y_pred, sample_weight=sample_weight,multioutput=multioutput
    # )
    output_errors = np.average(
        np.square(y_true - y_pred), weights=horizon_weight, axis=0
    )
    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    root_mean_squared_error
    root_median_squared_error

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
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is None:
        output_errors = np.median(np.abs(y_pred - y_true), axis=0)
    else:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
        output_errors = _weighted_percentile(
            np.abs(y_pred - y_true), sample_weight=horizon_weight
        )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def root_median_squared_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Root median squared error (RMdSE). RMdSE output is non-negative floating
    point. The best value is 0.0.

    Like RMSE, RMdSE is on same scale as the data. Because it squares the
    forecast error rather than taking the absolute value, it is more
    sensitive to outliers than MAE or MdAE.

    Taking the median instead of the mean of the squared errors also makes
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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        RMdSE loss.
        If multioutput is 'raw_values', then RMdSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average RMdSE of all output errors is returned.

    See Also
    --------
    mean_absolute_error
    median_absolute_error
    mean_squared_error
    median_squared_error
    root_mean_squared_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import root_median_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> root_median_squared_error(y_true, y_pred)
    0.5
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> root_median_squared_error(y_true, y_pred)
    0.7905694150420949
    >>> root_median_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.5, 1. ])
    >>> root_median_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.8803408430829504

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    mdse = median_squared_error(
        y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
    )
    return np.sqrt(mdse)


def median_squared_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Median squared error (MdSE). MdSE output is non-negative floating
    point. The best value is 0.0.

    Like MSE, MdSE is measured in squared units of the input data. Because it
    squares the forecast error rather than taking the absolute value, it is more
    sensitive to outliers than MAE or MdAE.

    Taking the median instead of the mean of the squared errors also makes
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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    root_mean_squared_error
    root_median_squared_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import median_squared_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_squared_error(y_true, y_pred)
    0.25
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_squared_error(y_true, y_pred)
    0.625
    >>> median_squared_error(y_true, y_pred, multioutput='raw_values')
    array([0.25, 1. ])
    >>> median_squared_error(y_true, y_pred, multioutput=[0.3, 0.7])
    0.7749999999999999

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is None:
        output_errors = np.median(np.square(y_pred - y_true), axis=0)
    else:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
        output_errors = _weighted_percentile(
            np.square(y_pred - y_true), sample_weight=horizon_weight
        )

    if isinstance(multioutput, str):
        if multioutput == "raw_values":
            return output_errors
        elif multioutput == "uniform_average":
            # pass None as weights to np.average: uniform mean
            multioutput = None

    return np.average(output_errors, weights=multioutput)


def symmetric_mean_absolute_percentage_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Symmetric mean absolute percentage error (sMAPE). sMAPE output is
    non-negative floating point. The best value is 0.0.

    sMAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it is less sensitive to outliers than RMSPE or RMdSPE.

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        sMAPE loss.
        If multioutput is 'raw_values', then sMAPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average sMAPE of all output errors is returned.

    See Also
    --------
    symmetric_median_absolute_percentage_error
    mean_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error
    root_mean_squared_percentage_error
    root_median_squared_percentage_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        symmetric_mean_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> symmetric_mean_absolute_percentage_error(y_true, y_pred)
    55.53379953379954
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> symmetric_mean_absolute_percentage_error(y_true, y_pred)
    55.53379953379954
    >>> symmetric_mean_absolute_percentage_error(
            y_true, y_pred, multioutput='raw_values'
        )
    array([71.11111111, 50.50505051])
    >>> symmetric_mean_absolute_percentage_error(
            y_true, y_pred, multioutput=[0.3, 0.7]
        )
    56.68686868686869

    Parameters
    ----------
    y_true : pandas Series of shape = (fh,) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pandas Series of shape = (fh,)
        Estimated target values.

    Returns
    -------
    loss : float
        sMAPE loss

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """

    return mean_absolute_percentage_error(
        y_true,
        y_pred,
        symmetric=True,
        horizon_weight=horizon_weight,
        multioutput=multioutput,
    )


def symmetric_median_absolute_percentage_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Symmetric median absolute percentage error (sMdAPE). sMdAPE output is
    non-negative floating point. The best value is 0.0.

    sMdAPE is measured in percentage error relative to the test data. Because it
    takes the absolute value rather than square the percentage forecast
    error, it is less sensitive to outliers than RMSPE or RMdSPE.

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        sMAPE loss.
        If multioutput is 'raw_values', then sMAPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average sMAPE of all output errors is returned.

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        symmetric_median_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> symmetric_median_absolute_percentage_error(y_true, y_pred)
    18.181818181818183
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> symmetric_median_absolute_percentage_error(y_true, y_pred)
    40.0
    >>> symmetric_median_absolute_percentage_error(
            y_true, y_pred, multioutput='raw_values'
        )
    array([13.33333333, 66.66666667])
    >>> symmetric_median_absolute_percentage_error(
            y_true, y_pred, multioutput=[0.3, 0.7]
        )
    50.666666666666664

    Parameters
    ----------
    y_true : pandas Series of shape = (fh,) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pandas Series of shape = (fh,)
        Estimated target values.

    Returns
    -------
    loss : float
        sMdAPE loss

    See Also
    --------
    symmetric_mean_absolute_percentage_error
    mean_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error
    root_mean_squared_percentage_error
    root_median_squared_percentage_error

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """

    return median_absolute_percentage_error(
        y_true,
        y_pred,
        symmetric=True,
        horizon_weight=horizon_weight,
        multioutput=multioutput,
    )


def mean_absolute_percentage_error(
    y_true,
    y_pred,
    symmetric=False,
    horizon_weight=None,
    multioutput="uniform_average",
):
    """Mean absolute percentage error (MAPE). MAPE output is non-negative floating
    point. The best value is 0.0.

    Symmetric = True calculates symmetric absolute percentage error (sMAPE).

    MAPE and sMAPE are measured in percentage error relative to the test data.
    Because they take the absolute value rather than square the percentage
    forecast error, they is less sensitive to outliers than RMSPE or RMdSPE.

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

    symmetric : bool, default = False
        Whether to calculate symmetric percentage error.

    horizon_weight : array-like of shape (fh,), default=None
        Forecast horizon weights.

    multioutput : {'raw_values', 'uniform_average'}  or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        MAPE unless symmetric is True, then sMAPE is returned.
        If multioutput is 'raw_values', then MAPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MAPE of all output errors is returned.

    See Also
    --------
    symmetric_mean_absolute_percentage_error
    symmetric_median_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error
    root_mean_squared_percentage_error
    root_median_squared_percentage_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        mean_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_absolute_percentage_error(y_true, y_pred)
    33.69047619047619
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_absolute_percentage_error(y_true, y_pred)
    55.15873015873015
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    array([38.0952381, 72.22222222])
    >>> mean_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    61.98412698412698

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """

    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is not None:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)

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


def root_mean_squared_percentage_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Root mean squared percentage error (RMSPE). RMSPE output is non-negative
    floating point. The best value is 0.0.

    RMSPE is measured in percentage error relative to the test data. Because it
    takes the square rather than absolute value of the percentage forecast
    error, it is more sensitive to outliers than MAPE or MdAPE.

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        RMSPE loss.
        If multioutput is 'raw_values', then RMSPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average RMSPE of all output errors is returned.

    See Also
    --------
    symmetric_mean_absolute_percentage_error
    symmetric_median_absolute_percentage_error
    mean_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error
    root_median_squared_percentage_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        root_mean_squared_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> root_mean_squared_percentage_error(y_true, y_pred)
    48.76086424671088
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> root_mean_squared_percentage_error(y_true, y_pred)
    71.27629270478701
    >>> root_mean_squared_percentage_error(y_true, y_pred, multioutput='raw_values')
    array([58.32118435, 82.21471437])
    >>> root_mean_squared_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    75.84121336547028

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.s
    """
    mspe = mean_squared_percentage_error(
        y_true, y_pred, horizon_weight=None, multioutput=multioutput
    )
    # Here we are multipying by 10 b/c mspe is already converted to "percentage"
    # units by multiplying by 100. Multipylying by 10 reverses the square root
    # of this multiplicative factor
    return np.sqrt(mspe)


def mean_squared_percentage_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Mean squared percentage error (MSPE). MSPE output is non-negative
    floating point. The best value is 0.0.

    MSPE is measured in squared percentage error relative to the test data.
    Because it takes the square rather than absolute value of the percentage
    forecast error, it is more sensitive to outliers than MAPE or MdAPE.

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        RMSPE loss.
        If multioutput is 'raw_values', then RMSPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average RMSPE of all output errors is returned.

    See Also
    --------
    symmetric_mean_absolute_percentage_error
    symmetric_median_absolute_percentage_error
    mean_absolute_percentage_error
    median_absolute_percentage_error
    median_squared_percentage_error
    root_mean_squared_percentage_error
    root_median_squared_percentage_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import mean_squared_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> mean_squared_percentage_error(y_true, y_pred)
    2377.6218820861677
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> mean_squared_percentage_error(y_true, y_pred)
    5080.309901738473
    >>> mean_squared_percentage_error(y_true, y_pred, multioutput='raw_values')
    array([3401.36054422, 6759.25925926])
    >>> mean_squared_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    5751.889644746787

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.s
    """
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is not None:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)

    output_errors = np.average(
        np.square(_percentage_error(y_true, y_pred)),
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
    y_true, y_pred, symmetric=False, horizon_weight=None, multioutput="uniform_average"
):
    """Median absolute percentage error (MdAPE). MdAPE output is non-negative
    floating point. The best value is 0.0.

    Symmetric = True calculates symmetric absolute percentage error (sMAPE).

    MdAPE and sMdAPE are measured in percentage error relative to the test data.
    Because they take the absolute value rather than square the percentage
    forecast error, they are less sensitive to outliers than RMSPE or RMdSPE.

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        MdAPE unless symmetric is True, then sMdAPE is returned.
        If multioutput is 'raw_values', then MdAPE or sMdAPE is returned for
        each output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdAPE or sMdAPE of all output errors is returned.

    See Also
    --------
    symmetric_mean_absolute_percentage_error
    symmetric_median_absolute_percentage_error
    mean_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error
    root_mean_squared_percentage_error
    root_median_squared_percentage_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        median_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_absolute_percentage_error(y_true, y_pred)
    16.666666666666668
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_absolute_percentage_error(y_true, y_pred)
    57.142857142857146
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    array([14.28571429, 100.])
    >>> median_absolute_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    74.28571428571429

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is None:
        output_errors = np.median(
            np.abs(_percentage_error(y_true, y_pred, symmetric=symmetric)), axis=0
        )
    else:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
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


def root_median_squared_percentage_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Root median squared percentage error (RMdSPE). RMdSPE output is
    non-negative floating point. The best value is 0.0.

    RMdSPE is measured in percentage error relative to the test data. Because it
    takes the square rather than absolute value of the percentage forecast
    error, it is more sensitive to outliers than MAPE or MdAPE.

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        RMdSPE loss.
        If multioutput is 'raw_values', then RMdSPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average RMdSPE of all output errors is returned.

    See Also
    --------
    symmetric_mean_absolute_percentage_error
    symmetric_median_absolute_percentage_error
    mean_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error
    root_mean_squared_percentage_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        root_median_squared_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> root_median_squared_percentage_error(y_true, y_pred)
    16.666666666666668
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> root_median_squared_percentage_error(y_true, y_pred)
    71.42857142857143
    >>> root_median_squared_percentage_error(y_true, y_pred, multioutput='raw_values')
    array([14.28571429, 100. ])
    >>> root_median_squared_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    84.03109239915852

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """

    mdspe = median_squared_percentage_error(
        y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
    )

    return np.sqrt(mdspe)


def median_squared_percentage_error(
    y_true, y_pred, horizon_weight=None, multioutput="uniform_average"
):
    """Median squared percentage error (MdSPE). MdSPE output is non-negative
    floating point. The best value is 0.0.

    MdSPE is measured in squared percentage error relative to the test data.
    Because it takes the square rather than absolute value of the percentage
    forecast error, it is more sensitive to outliers than MAPE or MdAPE.

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        MdSPE loss.
        If multioutput is 'raw_values', then MdSPE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average MdSPE of all output errors is returned.

    See Also
    --------
    symmetric_mean_absolute_percentage_error
    symmetric_median_absolute_percentage_error
    mean_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error
    root_mean_squared_percentage_error
    root_median_squared_percentage_error

    Examples
    --------
    >>> from sktime.performance_metrics.forecasting import \
        median_squared_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7, 2])
    >>> y_pred = np.array([2.5, 0.0, 2, 8, 1.25])
    >>> median_squared_percentage_error(y_true, y_pred)
    277.7777777777778
    >>> y_true = np.array([[0.5, 1], [-1, 1], [7, -6]])
    >>> y_pred = np.array([[0, 2], [-1, 2], [8, -5]])
    >>> median_squared_percentage_error(y_true, y_pred)
    5102.040816326531
    >>> median_squared_percentage_error(y_true, y_pred, multioutput='raw_values')
    array([204.08163265, 10000.])
    >>> median_squared_percentage_error(y_true, y_pred, multioutput=[0.3, 0.7])
    7061.224489795918

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.s
    """
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    if horizon_weight is not None:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
        output_errors = _weighted_percentile(
            np.square(_percentage_error(y_true, y_pred)),
            sample_weight=horizon_weight,
        )
    else:
        output_errors = np.median(np.square(_percentage_error(y_true, y_pred)), axis=0)

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    >>> mean_relative_absolute_error(
            y_true, y_pred, y_pred_benchmark, multioutput='raw_values'
        )
    array([0.51851852, 1.22222222])
    >>> mean_relative_absolute_error(
            y_true, y_pred, y_pred_benchmark, multioutput=[0.3, 0.7]
        )
    1.0111111111111108

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    y_pred_benchmark = check_y(
        y_pred_benchmark, enforce_univariate=False, allow_numpy=True
    )
    if horizon_weight is None:
        output_errors = np.mean(
            np.abs(_relative_error(y_true, y_pred, y_pred_benchmark)), axis=0
        )
    else:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    >>> median_relative_absolute_error(
            y_true, y_pred, y_pred_benchmark, multioutput='raw_values'
        )
    array([0.55555556, 0.83333333])
    >>> median_relative_absolute_error(
            y_true, y_pred, y_pred_benchmark, multioutput=[0.3, 0.7]
        )
    0.7499999999999999

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """
    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    y_pred_benchmark = check_y(
        y_pred_benchmark, enforce_univariate=False, allow_numpy=True
    )
    if horizon_weight is None:
        output_errors = np.median(
            np.abs(_relative_error(y_true, y_pred, y_pred_benchmark)), axis=0
        )
    else:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    .5578632807409556
    >>> geometric_mean_relative_absolute_error(
            y_true, y_pred, y_pred_benchmark, multioutput='raw_values'
        )
    array([4.97801163e-06 1.11572158e+00])
    >>> geometric_mean_relative_absolute_error(
            y_true, y_pred, y_pred_benchmark, multioutput=[0.3, 0.7]
        )
    0.7810066018326863

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """

    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    y_pred_benchmark = check_y(
        y_pred_benchmark, enforce_univariate=False, allow_numpy=True
    )
    relative_errors = np.abs(_relative_error(y_true, y_pred, y_pred_benchmark))
    if horizon_weight is None:
        output_errors = gmean(
            np.where(relative_errors == 0.0, EPS, relative_errors), axis=0
        )
    else:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
        output_errors = weighted_geometric_mean(
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
    y_true, y_pred, y_pred_benchmark, horizon_weight=None, multioutput="uniform_average"
):
    """Geometric mean relative squared error (GMRSE).

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    Returns
    -------
    loss : float
        GMRSE loss.
        If multioutput is 'raw_values', then GMRSE is returned for each
        output separately.
        If multioutput is 'uniform_average' or an ndarray of weights, then the
        weighted average GMRSE of all output errors is returned.

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
    >>> geometric_mean_relative_squared_error(
            y_true, y_pred, y_pred_benchmark, multioutput='raw_values'
        )
    array([04.09227746e-06, 1.24483465e+00])
    >>> geometric_mean_relative_squared_error(
            y_true, y_pred, y_pred_benchmark, multioutput=[0.3, 0.7]
        )
    0.8713854839582426

    References
    ----------
    ..[1]   Hyndman, R. J and Koehler, A. B. (2006).
            "Another look at measures of forecast accuracy", International
            Journal of Forecasting, Volume 22, Issue 4.
    """

    y_true, y_pred = check_y_true_pred(y_true, y_pred)
    y_pred_benchmark = check_y(
        y_pred_benchmark, enforce_univariate=False, allow_numpy=True
    )
    relative_errors = np.square(_relative_error(y_true, y_pred, y_pred_benchmark))
    if horizon_weight is None:
        output_errors = gmean(
            np.where(relative_errors == 0.0, EPS, relative_errors), axis=0
        )
    else:
        horizon_weight = check_horizon_weights(horizon_weight, y_pred)
        output_errors = weighted_geometric_mean(
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


def relative_loss(
    y_true,
    y_pred,
    y_pred_benchmark,
    loss_function=mean_absolute_error,
    horizon_weight=None,
    multioutput="uniform_average",
):
    """Calculates relative loss for a set of predictions and benchmark
    predictions for a given loss function

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
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

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
    return np.divide(
        loss_function(
            y_true, y_pred, horizon_weight=horizon_weight, multioutput=multioutput
        ),
        loss_function(
            y_true,
            y_pred_benchmark,
            horizon_weight=horizon_weight,
            multioutput=multioutput,
        ),
    )


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
    y_true, y_pred = check_y_true_pred(y_true, y_pred)

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


def _percentage_error(y_true, y_pred, symmetric=False):
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
            200
            * np.abs(y_true - y_pred)
            / np.maximum(np.abs(y_true) + np.abs(y_pred), EPS)
        )
    else:
        percentage_error = 100 * (y_true - y_pred) / np.maximum(np.abs(y_true), EPS)
    return percentage_error
