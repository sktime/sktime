#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating forecasting models."""

__author__ = ["aiwalter", "mloning"]
__all__ = ["evaluate"]

import time
import warnings

import numpy as np
import pandas as pd

from sktime.exceptions import FitFailedWarning
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation.forecasting import (
    check_cv,
    check_fh,
    check_scoring,
    check_X,
)
from sktime.utils.validation.series import check_series


def evaluate(
    forecaster,
    cv,
    y,
    X=None,
    strategy="refit",
    scoring=None,
    return_data=False,
    error_score=np.nan,
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
    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
    scoring : subclass of sktime.performance_metrics.BaseMetric, default=None.
        Used to get a score function that takes y_pred and y_test arguments
        and accept y_train as keyword argument.
        If None, then uses scoring = MeanAbsolutePercentageError(symmetric=True).
    return_data : bool, default=False
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    Returns
    -------
    pd.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.

    Examples
    --------
        The type of evaluation that is done by `evaluate` depends on metrics in
        param `scoring`. Default is `MeanAbsolutePercentageError`.
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="mean", sp=12)
    >>> cv = ExpandingWindowSplitter(initial_window=12, step_length=3,
    ... fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv)

        Optionally, users may select other metrics that can be supplied
        by `scoring` argument. These can be forecast metrics of any kind,
        i.e., point forecast metrics, interval metrics, quantile foreast metrics.
        https://www.sktime.org/en/stable/api_reference/performance_metrics.html?highlight=metrics

        To evaluate estimators using a specific metric, provide them to the scoring arg.
    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteError
    >>> loss = MeanAbsoluteError()
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv, scoring=loss)

        An example of an interval metric is the `PinballLoss`.
        It can be used with all probabilistic forecasters.
    >>> from sktime.forecasting.naive import NaiveVariance
    >>> from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
    >>> loss = PinballLoss()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> results = evaluate(forecaster=NaiveVariance(forecaster),
    ... y=y, cv=cv, scoring=loss)
    """
    _check_strategy(strategy)
    cv = check_cv(cv, enforce_start_with_window=True)
    scoring = check_scoring(scoring)
    y = check_series(
        y,
        enforce_univariate=forecaster.get_tag("scitype:y") == "univariate",
        enforce_multivariate=forecaster.get_tag("scitype:y") == "multivariate",
    )
    X = check_X(X)

    # Define score name.
    score_name = "test_" + scoring.name

    # Initialize dataframe.
    results = []

    # Run temporal cross-validation.
    for i, (train, test) in enumerate(cv.split(y)):

        # set default result values in case estimator fitting fails
        score = error_score
        fit_time = np.nan
        pred_time = np.nan
        cutoff = np.nan
        y_pred = np.nan

        # split data
        y_train, y_test, X_train, X_test = _split(y, X, train, test, cv.fh)

        # create forecasting horizon
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        try:
            # fit/update
            start_fit = time.perf_counter()
            if i == 0 or strategy == "refit":
                forecaster = forecaster.clone()
                forecaster.fit(y_train, X_train, fh=fh)

            else:  # if strategy in ["update", "no-update_params"]:
                update_params = strategy == "update"
                forecaster.update(y_train, X_train, update_params=update_params)
            fit_time = time.perf_counter() - start_fit

            pred_type = {
                "pred_quantiles": "forecaster.predict_quantiles",
                "pred_intervals": "forecaster.predict_interval",
                "pred_proba": "forecaster.predict_proba",
                None: "forecaster.predict",
            }
            # predict
            start_pred = time.perf_counter()

            if hasattr(scoring, "metric_args"):
                metric_args = scoring.metric_args

            try:
                scitype = scoring.get_tag("scitype:y_pred")
            except ValueError:
                # If no scitype exists then metric is not proba and no args needed
                scitype = None
                metric_args = {}

            y_pred = eval(pred_type[scitype])(fh, X_test, **metric_args)

            pred_time = time.perf_counter() - start_pred

            # score
            score = scoring(y_test, y_pred, y_train=y_train)

            # cutoff
            cutoff = forecaster.cutoff

        except Exception as e:
            if error_score == "raise":
                raise e
            else:
                warnings.warn(
                    f"""
                Fitting of forecaster failed, you can set error_score='raise' to see
                the exception message. Fit failed for len(y_train)={len(y_train)}.
                The score will be set to {error_score}.
                Failed forecaster: {forecaster}.
                """,
                    FitFailedWarning,
                )

        # save results
        results.append(
            {
                score_name: score,
                "fit_time": fit_time,
                "pred_time": pred_time,
                "len_train_window": len(y_train),
                "cutoff": cutoff,
                "y_train": y_train if return_data else np.nan,
                "y_test": y_test if return_data else np.nan,
                "y_pred": y_pred if return_data else np.nan,
            }
        )

    results = pd.DataFrame(results)
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
        # For test, we begin by returning the full range of test/train values.
        # for those transformers that change the size of input.
        test = np.arange(test[-1] + 1)
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
        must be in "refit", "update" , "no-update_params"

    Raises
    ------
    ValueError
        If strategy value is not in expected values, raise error.
    """
    valid_strategies = ("refit", "update", "no-update_params")
    if strategy not in valid_strategies:
        raise ValueError(f"`strategy` must be one of {valid_strategies}")
