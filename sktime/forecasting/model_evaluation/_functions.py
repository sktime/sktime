#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating forecasting models."""

__author__ = ["aiwalter", "mloning"]
__all__ = ["evaluate"]

import time

import numpy as np
import pandas as pd

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
    return_data : bool, default=False
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
        The type of evaluation that is done by `evaluate` depends on metrics in
        param `scoring`
        When evaluating model/estimators on point forecast, users can let
        scoring=None, which defaults to MeanAbsolutePercentageError
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

        To evaluate models/estimators using a specific metric, provide them to the
        scoring arg.
    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteError
    >>> loss = MeanAbsoluteError()
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv, scoring=loss)

        An example of an interval metric is the PinballLoss. It can be used with
        all probabilistic forecasters.
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
        # split data
        y_train, y_test, X_train, X_test = _split(y, X, train, test, cv.fh)

        # create forecasting horizon
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        # fit/update
        start_fit = time.perf_counter()
        if i == 0 or strategy == "refit":
            forecaster = forecaster.clone()
            forecaster.fit(y_train, X_train, fh=fh)

        else:  # if strategy == "update":
            forecaster.update(y_train, X_train)
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

        # save results
        results.append(
            {
                score_name: score,
                "fit_time": fit_time,
                "pred_time": pred_time,
                "len_train_window": len(y_train),
                "cutoff": forecaster.cutoff,
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
