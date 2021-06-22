# -*- coding: utf-8 -*-

__author__ = ["Martin Walter", "Markus Löning"]
__all__ = ["evaluate"]

import numpy as np
import pandas as pd
import time
from sktime.utils.validation.forecasting import check_y_X
from sktime.utils.validation.forecasting import check_cv
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation.forecasting import check_scoring, check_fh


def evaluate(
    forecaster,
    cv,
    y,
    X=None,
    strategy="refit",
    scoring=None,
    fit_params=None,
    return_data=False,
):
    """Evaluate forecaster using timeseries cross-validation.

    Parameters
    ----------
    forecaster : sktime.forecaster
        Any forecaster
    cv : Temporal cross-validation splitter
        Splitter of how to split the data into test data and train data
    y : pd.Series
        Target time series to which to fit the forecaster.
    X : pd.DataFrame, default=None
        Exogenous variables
    strategy : {"refit", "update"}
        Must be "refit" or "update". The strategy defines whether the `forecaster` is
        only fitted on the first train window data and then updated, or always refitted.
    scoring : subclass of sktime.performance_metrics.BaseMetric, default=None.
        Used to get a score function that takes y_pred and y_test arguments
        and accept y_train as keyword argument.
        If None, then uses scoring = MeanAbsolutePercentageError(symmetric=True).
    fit_params : dict, default=None
        Parameters passed to the `fit` call of the forecaster.
    return_data : bool, default=False
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.

    Returns
    ----------
    pd.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.

    Example
    -------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="mean", sp=12)
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=24,
    ...     step_length=12,
    ...     fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv)
    """
    _check_strategy(strategy)
    cv = check_cv(cv, enforce_start_with_window=True)
    scoring = check_scoring(scoring)
    y, X = check_y_X(y, X)
    fit_params = {} if fit_params is None else fit_params

    # Define score name.
    score_name = "test_" + scoring.name

    # Initialize dataframe.
    results = pd.DataFrame()

    # Run temporal cross-validation.
    for i, (train, test) in enumerate(cv.split(y)):
        # split data
        y_train, y_test, X_train, X_test = _split(y, X, train, test, cv.fh)

        # create forecasting horizon
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        # fit/update
        start_fit = time.time()
        if i == 0 or strategy == "refit":
            forecaster.fit(y_train, X_train, fh=fh, **fit_params)

        else:  # if strategy == "update":
            forecaster.update(y_train, X_train)
        fit_time = time.time() - start_fit

        # predict
        start_pred = time.time()
        y_pred = forecaster.predict(fh, X=X_test)
        pred_time = time.time() - start_pred

        # score
        score = scoring(y_test, y_pred, y_train=y_train)

        # save results
        results = results.append(
            {
                score_name: score,
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


def _split(y, X, train, test, fh):
    """Split y and X for given train and test set indices."""
    y_train = y.iloc[train]
    y_test = y.iloc[test]

    cutoff = y_train.index[-1]
    fh = check_fh(fh)
    fh = fh.to_relative(cutoff)

    if X is not None:
        X_train = X.iloc[train, :]

        # We need to expand test indices to a full range, since some forecasters
        # require the full range of exogenous values.
        test = np.arange(test[0] - fh.min(), test[-1]) + 1
        X_test = X.iloc[test, :]
    else:
        X_train = None
        X_test = None

    return y_train, y_test, X_train, X_test


def _check_strategy(strategy):
    """Assert strategy value.

    Parameters
    ----------
    strategy : str
        strategy of how to evaluate a forecaster

    Raises
    ------
    ValueError
        If strategy value is not in expected values, raise error.
    """
    valid_strategies = ("refit", "update")
    if strategy not in valid_strategies:
        raise ValueError(f"`strategy` must be one of {valid_strategies}")
