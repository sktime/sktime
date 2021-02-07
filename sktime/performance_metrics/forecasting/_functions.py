#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import time
from typing import types
from tqdm.auto import tqdm

from sktime.utils.validation.series import check_equal_time_index
from sktime.utils.validation.series import check_time_index
from sktime.utils.validation.forecasting import check_y
from sktime.utils.validation.forecasting import check_cv
from sktime.forecasting.base import ForecastingHorizon


__author__ = ["Markus Löning", "Tomasz Chodakowski", "Martin Walter"]
__all__ = ["mase_loss", "smape_loss", "mape_loss", "evaluate"]


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


def evaluate(
    forecaster, cv, y, X=None, strategy="refit", scoring=smape_loss, return_data=False
):
    """Evaluate forecaster using cross-validation

    Parameters
    ----------
    forecaster : sktime.forecaster
        Any forecaster
    cv : sktime.SlidingWindowSplitter or sktime.ExpandingWindowSplitter
        Splitter of how to split the data into test data and train data
    y : pd.Series
        Target time series to which to fit the forecaster.
    X : pd.DataFrame, optional (default=None)
        Exogenous variables
    strategy : str, optional
        Must be "refit" or "update", by default "refit". The strategy defines
        whether forecaster is only fitted on the first train window data and
        then updated or always refitted.
    scoring : function, optional
        A score function that takes y_pred and y_test as arguments,
        by default smape_loss
    return_data : bool, optional
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.

    Returns
    -------
    pd.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.performance_metrics.forecasting import evaluate
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="drift", sp=12)
    >>> cv = ExpandingWindowSplitter(
        initial_window=24,
        step_length=12,
        fh=[1,2,3,4,5,6,7,8,9,10,11,12]
        )
    >>> evaluate(forecaster=forecaster, y=y, cv=cv)
    """
    check_cv(cv)
    _check_strategies(strategy)
    assert cv.initial_window is not None, "cv must have an initial_window"
    assert isinstance(evaluate, types.FunctionType), "scoring must be a function"

    n_splits = cv.get_n_splits(y)
    results = pd.DataFrame()

    for i, (train, test) in enumerate(tqdm(cv.split(y), total=n_splits)):
        # workaroud to avoid training on smaller windows
        if len(train) >= cv.initial_window:

            # create train/test data
            y_train = y.iloc[train]
            y_test = y.iloc[test]

            X_train = X.iloc[train] if X else None
            X_test = X.iloc[test] if X else None

            # fit/update
            start_fit = time.time()
            if strategy == "refit" or i == 0:
                forecaster.fit(
                    y=y_train,
                    X=X_train,
                    fh=ForecastingHorizon(y_test.index, is_relative=False),
                )
            else:
                # strategy == "update" and i != 0:
                forecaster.update(y=y_train, X=X_train)
            fit_time = time.time() - start_fit

            # predict
            start_pred = time.time()
            y_pred = forecaster.predict(
                fh=ForecastingHorizon(y_test.index, is_relative=False), X=X_test
            )
            pred_time = time.time() - start_pred

            # save results
            results = results.append(
                {
                    "test_" + scoring.__name__: scoring(y_pred, y_test),
                    "fit_time": fit_time,
                    "pred_time": pred_time,
                    "len_train_window": len(y_train),
                    "cutoff": forecaster.cutoff,
                    "y_train": y_train if return_data else np.nan,
                    "y_test": y_test if return_data else np.nan,
                    "y_pred": y_pred if return_data else np.nan,
                },
                ignore_index=True,
            )

    # post-processing of results
    if not return_data:
        results = results.drop(columns=["y_train", "y_test", "y_pred"])
    results["len_train_window"] = results["len_train_window"].astype(int)

    return results


def _check_strategies(strategy):
    """Assert strategy value

    Parameters
    ----------
    strategy : str
        strategy of how to evaluate a forecaster

    Raises
    ------
    ValueError
        If strategy value is not in expected values, raise error.
    """
    if strategy not in ["refit", "update"]:
        raise ValueError('strategy must be either "refit" or "update"')
