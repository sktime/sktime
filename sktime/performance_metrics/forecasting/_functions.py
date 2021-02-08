#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np

from sktime.utils.validation.series import check_equal_time_index
from sktime.utils.validation.series import check_time_index
from sktime.utils.validation.forecasting import check_y


__author__ = ["Markus Löning", "Tomasz Chodakowski"]
__all__ = ["mase_loss", "smape_loss", "mape_loss"]


def mase_loss(y_test, y_pred, y_train, sp=1):
    """Mean absolute scaled error.

    This scale-free error metric can be used to compare forecast methods on
    a single
    series and also to compare forecast accuracy between series. This metric
    is well
    suited to intermittent-demand series because it never gives infinite or
    undefined
    values.

    Parameters
    ----------
    y_test : pandas Series of shape = (fh,) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pandas Series of shape = (fh,)
        Estimated target values.
    y_train : pandas Series of shape = (n_obs,)
        Observed training values.
    sp : int
        Seasonal periodicity of training data.

    Returns
    -------
    loss : float
        MASE loss

    References
    ----------
    ..[1]   Hyndman, R. J. (2006). "Another look at measures of forecast
            accuracy", Foresight, Issue 4.
    """
    # input checks
    y_test = check_y(y_test)
    y_pred = check_y(y_pred)
    y_train = check_y(y_train)
    check_equal_time_index(y_test, y_pred)

    # check if training set is prior to test set
    if y_train is not None:
        check_time_index(y_train.index)
        if y_train.index.max() >= y_test.index.min():
            raise ValueError(
                "Found `y_train` with time index which is not "
                "before time index of `y_test`"
            )

    #  naive seasonal prediction
    y_train = np.asarray(y_train)
    y_pred_naive = y_train[:-sp]

    # mean absolute error of naive seasonal prediction
    mae_naive = np.mean(np.abs(y_train[sp:] - y_pred_naive))

    # if training data is flat, mae may be zero,
    # return np.nan to avoid divide by zero error
    # and np.inf values
    if mae_naive == 0:
        return np.nan
    else:
        return np.mean(np.abs(y_test - y_pred)) / mae_naive


def smape_loss(y_test, y_pred):
    """Symmetric mean absolute percentage error

    Parameters
    ----------
    y_test : pandas Series of shape = (fh,) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pandas Series of shape = (fh,)
        Estimated target values.

    Returns
    -------
    loss : float
        sMAPE loss
    """
    y_test = check_y(y_test)
    y_pred = check_y(y_pred)
    check_equal_time_index(y_test, y_pred)

    nominator = np.abs(y_test - y_pred)
    denominator = np.abs(y_test) + np.abs(y_pred)
    return np.mean(2.0 * nominator / denominator)


def mape_loss(y_test, y_pred):
    """Mean absolute percentage error (MAPE)
        MAPE output is non-negative floating point where the best value is 0.0.
        There is no limit on how large the error can be, particulalrly when `y_test`
        values are close to zero. In such cases the function returns a large value
        instead of `inf`.

    Parameters
    ----------
    y_test : pandas Series of shape = (fh,) where fh is the forecasting horizon
        Ground truth (correct) target values.
    y_pred : pandas Series of shape = (fh,)
        Estimated target values.

    Returns
    -------
    loss : float
        MAPE loss expressed as a fractional number rather than percentage point.


    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_error
    >>> y_test = pd.Series([1, -1, 2])
    >>> y_pred = pd.Series([2, -2, 4])
    >>> mape_loss(y_test, y_pred)
    1.0
    """

    y_test = check_y(y_test)
    y_pred = check_y(y_pred)
    check_equal_time_index(y_test, y_pred)

    eps = np.finfo(np.float64).eps

    return np.mean(np.abs(y_test - y_pred) / np.maximum(np.abs(y_test), eps))
