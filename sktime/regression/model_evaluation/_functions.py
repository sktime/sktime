#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utilities for performance evaluation of time series regression models."""

__author__ = ["Omswastik-11"]
__all__ = ["evaluate"]

import collections.abc
import inspect
import time
import warnings

import numpy as np
import pandas as pd

from sktime.datatypes import check_is_scitype, convert
from sktime.exceptions import FitFailedWarning
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.parallel import parallelize

PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


def _check_scores(metrics) -> list[collections.abc.Callable]:
    """Validate regression metrics.

    Parameters
    ----------
    metrics : sklearn metric function, list of sklearn metric functions, or None
        The regression metrics to validate.

    Returns
    -------
    metrics_list : list of callables
        Metrics coerced to a list and validated for callability.
    """
    if metrics is None:
        from sklearn.metrics import r2_score

        metrics = [r2_score]

    if not isinstance(metrics, list):
        metrics = [metrics]

    for metric in metrics:
        if not callable(metric):
            raise ValueError(f"Metric {metric} is not callable")

    return metrics


def _get_column_order_and_datatype(
    metrics, return_data: bool, include_sample_weight: bool
):
    """Get ordered columns and dtypes for results DataFrame."""
    metadata = {"fit_time": "float", "pred_time": "float"}
    for metric in metrics:
        metadata[f"test_{metric.__name__}"] = "float"

    if return_data:
        metadata.update(
            {
                "X_train": "object",
                "X_test": "object",
                "y_train": "object",
                "y_test": "object",
                "y_pred": "object",
            }
        )
        if include_sample_weight:
            metadata["sample_weight_train"] = "object"
            metadata["sample_weight_test"] = "object"

    return metadata.copy()


def _evaluate_fold(x, meta):
    i, (y_train, y_test, X_train, X_test, sw_train, sw_test) = x
    regressor = meta["regressor"]
    scoring = meta["scoring"]
    multioutput = meta["multioutput"]
    return_data = meta["return_data"]
    error_score = meta["error_score"]

    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    y_pred = pd.NA
    temp_result = dict()

    try:
        # fit
        start_fit = time.perf_counter()

        regressor = regressor.clone()
        fit_kwargs = {}
        if (
            sw_train is not None
            and "sample_weight" in inspect.signature(regressor.fit).parameters
        ):
            fit_kwargs["sample_weight"] = sw_train
        regressor.fit(y=y_train, X=X_train, **fit_kwargs)

        fit_time = time.perf_counter() - start_fit

        # predict
        start_pred = time.perf_counter()
        y_pred = regressor.predict(X_test)
        pred_time = time.perf_counter() - start_pred
        temp_result["pred_time"] = [pred_time]

        # score
        for metric in scoring:
            metric_kwargs = {}
            metric_sig = inspect.signature(metric).parameters
            if sw_test is not None and "sample_weight" in metric_sig:
                metric_kwargs["sample_weight"] = sw_test
            if multioutput is not None and "multioutput" in metric_sig:
                metric_kwargs["multioutput"] = multioutput

            score = metric(y_test, y_pred, **metric_kwargs)
            temp_result[f"test_{metric.__name__}"] = [score]

    except Exception as e:
        if error_score == "raise":
            raise e
        else:
            temp_result["pred_time"] = [pred_time]
            for metric in scoring:
                temp_result[f"test_{metric.__name__}"] = [score]
            warnings.warn(
                f"""
                In evaluate, fitting of regressor {type(regressor).__name__} failed,
                you can set error_score='raise' in evaluate to see
                the exception message.
                Fit failed for the {i}-th data split, on training data y_train
                , and len(y_train)={len(y_train)}.
                The score will be set to {error_score}.
                Failed regressor with parameters: {regressor}.
                """,
                FitFailedWarning,
                stacklevel=2,
            )

    temp_result["fit_time"] = [fit_time]

    if return_data:
        temp_result.update(
            {
                "X_train": [X_train],
                "X_test": [X_test],
                "y_train": [y_train],
                "y_test": [y_test],
                "y_pred": [y_pred],
            }
        )
        if sw_train is not None:
            temp_result["sample_weight_train"] = [sw_train]
        if sw_test is not None:
            temp_result["sample_weight_test"] = [sw_test]

    result = pd.DataFrame(temp_result)
    column_order = _get_column_order_and_datatype(
        scoring, return_data=return_data, include_sample_weight=sw_train is not None
    )
    result = result.reindex(columns=column_order.keys())

    return result


def evaluate(
    regressor,
    cv=None,
    X=None,
    y=None,
    scoring: collections.abc.Callable | list[collections.abc.Callable] | None = None,
    sample_weight=None,
    multioutput: str | np.ndarray | None = None,
    return_data: bool = False,
    error_score: str | int | float = np.nan,
    backend: str | None = None,
    backend_params: dict | None = None,
):
    r"""
    Evaluate regressor using time series cross-validation.

    All-in-one performance benchmarking utility for regressors, supporting
    optional sample weights and multioutput targets.

    Parameters
    ----------
    regressor : sktime.BaseRegressor
        Concrete sktime regressor to benchmark.

    cv : int, sklearn cross-validation generator or an iterable, default=3-fold CV
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None = default = ``KFold(n_splits=3, shuffle=True)``
        - integer, number of folds folds in a ``KFold`` splitter, ``shuffle=True``
        - An iterable yielding (train, test) splits as arrays of indices.

    X : sktime-compatible panel data (Panel scitype)
        Panel data container.

    y : sktime-compatible tabular data (Table scitype)
        Target variable, 1D or 2D for multioutput.

    scoring : callable or list of callables, optional (default=None)
        A scoring function or list of scoring functions that take
        ``(y_true, y_pred)`` and optionally ``sample_weight``/``multioutput``.
        If None, defaults to ``r2_score``.

    sample_weight : array-like, optional
        Sample weights aligned with ``y``. Passed to metrics that accept
        ``sample_weight`` and to ``fit`` if the regressor supports it.

    multioutput : str, array-like or None, optional
        Multioutput handling strategy forwarded to metrics that accept a
        ``multioutput`` argument, e.g., ``"uniform_average"``.

    return_data : bool, default=False
        If True, adds columns ``y_train``, ``y_test``, ``y_pred`` (and weights
        if provided) to the result.

    error_score : {"raise"} or float, default=np.nan
        Value to assign if estimator fitting fails.
        If "raise", the exception is raised.
        If a numeric value, a ``FitFailedWarning`` is raised and the value is
        assigned.

    backend : {"dask", "dask_lazy", "loky", "multiprocessing", "threading",
               "joblib"}, default=None
        Enables parallel evaluation, see classification evaluate for details.

    backend_params : dict, optional
        Additional parameters passed to the backend.

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame indexed by fold with columns for scores, timings, and
        optionally data/weights.
    """
    if backend in ["dask", "dask_lazy"]:
        if not _check_soft_dependencies("dask", severity="none"):
            raise RuntimeError(
                "running evaluate with backend='dask' requires the dask package "
                "installed, but dask is not present in the python environment"
            )

    scoring = _check_scores(scoring)

    # default handling for cv
    if isinstance(cv, int):
        from sklearn.model_selection import KFold

        _cv = KFold(n_splits=cv, shuffle=True)
    elif cv is None:
        from sklearn.model_selection import KFold

        _cv = KFold(n_splits=3, shuffle=True)
    else:
        _cv = cv

    y_valid, _, y_metadata = check_is_scitype(y, scitype="Table", return_metadata=[])
    if not y_valid:
        raise TypeError(f"Expected y dtype Table. Got {type(y)} instead.")
    y_mtype = y_metadata.get("mtype", None)
    y = convert(y, from_type=y_mtype, to_type="pd_DataFrame_Table")

    X_valid, _, X_metadata = check_is_scitype(X, scitype="Panel", return_metadata=[])
    if not X_valid:
        raise TypeError(f"Expected X dtype Panel. Got {type(X)} instead.")
    X_mtype = X_metadata.get("mtype", None)
    X = convert(X, from_type=X_mtype, to_type=PANDAS_MTYPES)

    if sample_weight is not None:
        if len(sample_weight) != len(y):
            raise ValueError("sample_weight length must match y")
        sample_weight = pd.Series(sample_weight, index=y.index)

    _evaluate_fold_kwargs = {
        "regressor": regressor,
        "scoring": scoring,
        "multioutput": multioutput,
        "return_data": return_data,
        "error_score": error_score,
    }

    def gen_y_X_train_test(y, X, cv, sample_weight=None):
        """Generate joint splits of y, X, and sample weights as per cv."""
        instance_idx = X.index.get_level_values(0).unique()

        for train_instance_idx, test_instance_idx in cv.split(instance_idx):
            train_instances = instance_idx[train_instance_idx]
            test_instances = instance_idx[test_instance_idx]

            X_train = X.loc[X.index.get_level_values(0).isin(train_instances)]
            X_test = X.loc[X.index.get_level_values(0).isin(test_instances)]

            y_train = y.iloc[train_instance_idx]
            y_test = y.iloc[test_instance_idx]

            sw_train = None
            sw_test = None
            if sample_weight is not None:
                sw_train = sample_weight.iloc[train_instance_idx]
                sw_test = sample_weight.iloc[test_instance_idx]

            yield y_train, y_test, X_train, X_test, sw_train, sw_test

    yx_splits = gen_y_X_train_test(y, X, _cv, sample_weight=sample_weight)

    results = parallelize(
        fun=_evaluate_fold,
        iter=enumerate(yx_splits),
        meta=_evaluate_fold_kwargs,
        backend=backend,
        backend_params=backend_params,
    )

    if backend == "dask_lazy":
        import dask.dataframe as dd

        metadata = _get_column_order_and_datatype(
            scoring, return_data, include_sample_weight=sample_weight is not None
        )

        results = dd.from_delayed(results, meta=metadata)
    elif backend == "dask":
        results = pd.concat(list(results))
    else:
        results = pd.concat(results)

    results = results.reset_index(drop=True)

    return results
