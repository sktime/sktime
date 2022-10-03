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


def _evaluate_window(
    y,
    X,
    train,
    test,
    i,
    fh,
    forecaster,
    strategy,
    scoring,
    return_data,
    error_score,
    score_name,
):

    # set default result values in case estimator fitting fails
    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    cutoff = np.nan
    y_pred = np.nan

    # split data
    y_train, y_test, X_train, X_test = _split(y, X, train, test, fh)

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
        # get cutoff
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

    result = pd.DataFrame(
        {
            score_name: [score],
            "fit_time": [fit_time],
            "pred_time": [pred_time],
            "len_train_window": [len(y_train)],
            "cutoff": [cutoff],
            "y_train": [y_train if return_data else np.nan],
            "y_test": [y_test if return_data else np.nan],
            "y_pred": [y_pred if return_data else np.nan],
        }
    )

    # Return forecaster if "update"
    if strategy == "update":
        return result, forecaster
    else:
        return result


def evaluate(
    forecaster,
    cv,
    y,
    X=None,
    strategy="refit",
    scoring=None,
    return_data=False,
    error_score=np.nan,
    backend=None,
    compute=True,
    **kwargs,
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
    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
        Runs parallel evaluate if specified.
    compute : bool, default=True
        Only applied if backend is to set "dask". If set to True, returns
    **kwargs : Keyword arguments
        Only relevant if backend is specified. Additional kwargs are passed into
        `dask.distributed.get_client` or `dask.distributed.Client` if backend is
        set to "dask", otherwise kwargs are passed into `joblib.Parallel`

    Returns
    -------
    pd.DataFrame or dd.DataFrame
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

    score_name = "test_" + scoring.name
    _evaluate_window_kwargs = {
        "fh": cv.fh,
        "forecaster": forecaster,
        "scoring": scoring,
        "strategy": strategy,
        "return_data": return_data,
        "error_score": error_score,
        "score_name": score_name,
    }

    if backend is None or strategy == "update":
        # Run temporal cross-validation sequentially
        results = []
        for i, (train, test) in enumerate(cv.split(y)):
            if strategy == "update":
                result, forecaster = _evaluate_window(
                    y,
                    X,
                    train,
                    test,
                    i,
                    **_evaluate_window_kwargs,
                )
                _evaluate_window_kwargs["forecaster"] = forecaster
            else:
                result = _evaluate_window(
                    y,
                    X,
                    train,
                    test,
                    i,
                    **_evaluate_window_kwargs,
                )
            results.append(result)
        results = pd.concat(results)

    elif backend == "dask":
        # Use Dask delayed instead of joblib,
        # which uses Futures under the hood
        import dask.dataframe as dd
        from dask import delayed as dask_delayed

        results = []
        for i, (train, test) in enumerate(cv.split(y)):
            results.append(
                dask_delayed(_evaluate_window)(
                    y,
                    X,
                    train,
                    test,
                    i,
                    **_evaluate_window_kwargs,
                )
            )
        results = dd.from_delayed(
            results,
            meta={
                score_name: "float",
                "fit_time": "float",
                "pred_time": "float",
                "len_train_window": "int",
                "cutoff": "datetime64[ns]",
                "y_train": "object" if return_data else "float",
                "y_test": "object" if return_data else "float",
                "y_pred": "object" if return_data else "float",
            },
        )
        if compute:
            results = results.compute()

    else:
        # Otherwise use joblib
        from joblib import Parallel, delayed

        results = Parallel(**kwargs)(
            delayed(_evaluate_window)(
                y,
                X,
                train,
                test,
                i,
                **_evaluate_window_kwargs,
            )
            for i, (train, test) in enumerate(cv.split(y))
        )
        results = pd.concat(results)

    if not return_data:
        results = results.drop(columns=["y_train", "y_test", "y_pred"])
    results = results.astype({"len_train_window": int}).reset_index(drop=True)

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
