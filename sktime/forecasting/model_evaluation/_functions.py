#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating forecasting models."""

__author__ = ["aiwalter", "mloning", "fkiraly", "topher-lo"]
__all__ = ["evaluate"]

import time
import warnings

import numpy as np
import pandas as pd

from sktime.datatypes import check_is_scitype, convert_to
from sktime.exceptions import FitFailedWarning
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation._dependencies import _check_soft_dependencies
from sktime.utils.validation.forecasting import check_cv, check_scoring

PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


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


def _split(
    y,
    X,
    train,
    test,
    freq=None,
):
    # split data according to cv
    y_train, y_test = y.iloc[train], y.iloc[test]
    X_train, X_test = None, None

    if X is not None:
        # For X_test, we select the full range of test/train values.
        # for those transformers that change the size of input.
        test_plus_train = np.append(train, test)
        X_train, X_test = (
            X.iloc[train].sort_index(),
            X.iloc[test_plus_train].sort_index(),
        )  # Defensive sort

    # Defensive assignment of freq
    if freq is not None:
        try:
            if y_train.index.nlevels == 1:
                y_train.index.freq = freq
                y_test.index.freq = freq
            else:
                # See: https://github.com/pandas-dev/pandas/issues/33647
                y_train.index.levels[-1].freq = freq
                y_test.index.levels[-1].freq = freq
        except AttributeError:  # Can't set attribute for range or period index
            pass

        if X is not None:
            try:
                if X.index.nlevels == 1:
                    X_train.index.freq = freq
                    X_test.index.freq = freq
                else:
                    X_train.index.levels[-1].freq = freq
                    X_test.index.levels[-1].freq = freq
            except AttributeError:  # Can't set attribute for range or period index
                pass

    return y_train, y_test, X_train, X_test


def _select_fh_from_y(y):
    # create forecasting horizon
    # if cv object has fh, we use that
    idx = y.index
    # otherwise, if y_test is not hierarchical, we simply take the index of y_test
    if y.index.nlevels == 1:
        fh = ForecastingHorizon(idx, is_relative=False)
    # otherwise, y_test is hierarchical, and we take its unique time indices
    else:
        fh_idx = idx.get_level_values(-1).unique()
        fh = ForecastingHorizon(fh_idx, is_relative=False)
    return fh


def _evaluate_window(
    y,
    X,
    train,
    test,
    i,
    fh,
    freq,
    forecaster,
    strategy,
    scoring,
    return_data,
    score_name,
    error_score,
    cutoff_dtype,
):

    # set default result values in case estimator fitting fails
    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    cutoff = pd.Period(pd.NaT) if cutoff_dtype.startswith("period") else pd.NA
    y_pred = pd.NA

    # split data
    y_train, y_test, X_train, X_test = _split(
        y=y, X=X, train=train, test=test, freq=freq
    )
    if fh is None:
        fh = _select_fh_from_y(y_test)

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
            "y_train": [y_train if return_data else pd.NA],
            "y_test": [y_test if return_data else pd.NA],
            "y_pred": [y_pred if return_data else pd.NA],
        }
    ).astype({"cutoff": cutoff_dtype})

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
        sktime forecaster (concrete BaseForecaster descendant)
    cv : Temporal cross-validation splitter
        Splitter of how to split the data into test data and train data
    y : sktime time series container
        Target (endogeneous) time series used in the evaluation experiment
    X : sktime time series container, of same mtype as y
        Exogenous time series used in the evaluation experiment
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
        Runs parallel evaluate if specified and `strategy` is set as "refit".
        - "loky", "multiprocessing" and "threading": uses `joblib` Parallel loops
        - "dask": uses `dask`, requires `dask` package in environment
        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (`cloudpickle`) for "dask" and "loky" is generally more robust than the
        standard `pickle` library used in "multiprocessing".
    compute : bool, default=True
        If backend="dask", whether returned DataFrame is computed.
        If set to True, returns `pd.DataFrame`, otherwise `dask.dataframe.DataFrame`.
    **kwargs : Keyword arguments
        Only relevant if backend is specified. Additional kwargs are passed into
        `dask.distributed.get_client` or `dask.distributed.Client` if backend is
        set to "dask", otherwise kwargs are passed into `joblib.Parallel`.

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.
        Row index is splitter index of train/test fold in `cv`.
        Entries in the i-th row are for the i-th train/test split in `cv`.
        Columns are as follows:
        - test_{scoring.name}: (float) Model performance score.
        - fit_time: (float) Time in sec for `fit` or `update` on train fold.
        - pred_time: (float) Time in sec to `predict` from fitted estimator.
        - len_train_window: (int) Length of train window.
        - cutoff: (int, pd.Timestamp, pd.Period) cutoff = last time index in train fold.
        - y_train: (pd.Series) only present if see `return_data=True`
          train fold of the i-th split in `cv`, used to fit/update the forecaster.
        - y_pred: (pd.Series) present if see `return_data=True`
          forecasts from fitted forecaster for the i-th test fold indices of `cv`.
        - y_test: (pd.Series) present if see `return_data=True`
          testing fold of the i-th split in `cv`, used to compute the metric.

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
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        raise RuntimeError(
            "running evaluate with backend='dask' requires the dask package installed,"
            "but dask is not present in the python environment"
        )

    _check_strategy(strategy)
    cv = check_cv(cv, enforce_start_with_window=True)
    scoring = check_scoring(scoring)

    ALLOWED_SCITYPES = ["Series", "Panel", "Hierarchical"]

    y_valid, _, _ = check_is_scitype(y, scitype=ALLOWED_SCITYPES, return_metadata=True)
    if not y_valid:
        raise TypeError(
            f"Expected y dtype {ALLOWED_SCITYPES!r}. Got {type(y)} instead."
        )

    y = convert_to(y, to_type=PANDAS_MTYPES)

    freq = None
    try:
        if y.index.nlevels == 1:
            freq = y.index.freq
        else:
            freq = y.index.levels[0].freq
    except AttributeError:
        pass

    if X is not None:
        X_valid, _, _ = check_is_scitype(
            X, scitype=ALLOWED_SCITYPES, return_metadata=True
        )
        if not X_valid:
            raise TypeError(
                f"Expected X dtype {ALLOWED_SCITYPES!r}. Got {type(X)} instead."
            )
        X = convert_to(X, to_type=PANDAS_MTYPES)

    score_name = f"test_{scoring.name}"
    cutoff_dtype = str(y.index.dtype)
    _evaluate_window_kwargs = {
        "fh": cv.fh,
        "freq": freq,
        "forecaster": forecaster,
        "scoring": scoring,
        "strategy": strategy,
        "return_data": return_data,
        "error_score": error_score,
        "score_name": score_name,
        "cutoff_dtype": cutoff_dtype,
    }

    if backend is None or strategy in ["update", "no-update_params"]:
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
                "cutoff": cutoff_dtype,
                "y_train": "object",
                "y_test": "object",
                "y_pred": "object",
            },
        )
        if compute:
            results = results.compute()

    else:
        # Otherwise use joblib
        from joblib import Parallel, delayed

        results = Parallel(backend=backend, **kwargs)(
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
