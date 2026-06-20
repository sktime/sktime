#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Shared evaluate utilities for panel-data estimators (classification, regression).

Common logic for cross-validation evaluation of time-series classification
and regression estimators.  Both ``classification.model_evaluation.evaluate``
and ``regression.model_evaluation.evaluate`` delegate to ``_run_evaluate``
defined here, keeping their public API unchanged while eliminating code
duplication.
"""

__author__ = ["ksharma6", "jgyasu", "NAME-ASHWANIYADAV"]

import inspect
import time
import warnings

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datatypes import check_is_scitype, convert
from sktime.exceptions import FitFailedWarning
from sktime.utils.parallel import parallelize

PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


def _get_pred_args_from_metric(scitype, metric):
    """Return prediction keyword arguments for a metric, if any.

    For ``pred_quantiles`` metrics the ``alpha`` value is extracted;
    otherwise an empty dict is returned.
    """
    pred_args = {
        "pred_quantiles": "alpha",
    }
    if scitype in pred_args.keys():
        val = getattr(metric, pred_args[scitype], None)
        if val is not None:
            return {pred_args[scitype]: val}
    return {}


def _get_column_order_and_datatype(
    metric_types: dict, return_data: bool = True
) -> dict:
    """Get the ordered column name and input datatype of results."""
    y_metadata = {
        "X_train": "object",
        "X_test": "object",
        "y_train": "object",
        "y_test": "object",
    }
    fit_metadata, metrics_metadata = {"fit_time": "float"}, {}
    for scitype in metric_types:
        for metric in metric_types.get(scitype):
            pred_args = _get_pred_args_from_metric(scitype, metric)
            if pred_args == {}:
                time_key = f"{scitype}_time"
                result_key = f"test_{metric.__name__}"
                y_pred_key = f"y_{scitype}"
            else:
                argval = list(pred_args.values())[0]
                time_key = f"{scitype}_{argval}_time"
                result_key = f"test_{metric.__name__}_{argval}"
                y_pred_key = f"y_{scitype}_{argval}"
            fit_metadata[time_key] = "float"
            metrics_metadata[result_key] = "float"
            if return_data:
                y_metadata[y_pred_key] = "object"
    if return_data:
        fit_metadata.update(y_metadata)
    metrics_metadata.update(fit_metadata)
    return metrics_metadata.copy()


def _evaluate_fold(x, meta):
    """Evaluate a single CV fold for a panel-data estimator.

    Parameters
    ----------
    x : tuple of (int, tuple)
        ``(fold_index, (y_train, y_test, X_train, X_test))``.
    meta : dict
        Must contain keys ``"estimator"``, ``"estimator_type"``,
        ``"scoring"``, ``"return_data"``, and ``"error_score"``.

    Returns
    -------
    result : pd.DataFrame
        Single-row DataFrame with scores, timings, and optionally data.
    """
    i, (y_train, y_test, X_train, X_test) = x
    estimator = meta["estimator"]
    estimator_type = meta["estimator_type"]
    scoring = meta["scoring"]
    return_data = meta["return_data"]
    error_score = meta["error_score"]

    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    y_pred = pd.NA
    temp_result = dict()
    y_preds_cache = dict()

    try:
        # fit
        start_fit = time.perf_counter()

        estimator = estimator.clone()
        estimator.fit(y=y_train, X=X_train)

        fit_time = time.perf_counter() - start_fit

        # predict based on metrics
        pred_type = {
            "pred_quantiles": "predict_quantiles",
            "pred_proba": "predict_proba",
            "pred": "predict",
        }
        # cache prediction from the first scitype and reuse it to
        # compute other metrics
        for scitype in scoring:
            method = getattr(estimator, pred_type[scitype])
            for metric in scoring.get(scitype):
                pred_args = _get_pred_args_from_metric(scitype, metric)
                if pred_args == {}:
                    time_key = f"{scitype}_time"
                    result_key = f"test_{metric.__name__}"
                    y_pred_key = f"y_{scitype}"
                else:
                    argval = list(pred_args.values())[0]
                    time_key = f"{scitype}_{argval}_time"
                    result_key = f"test_{metric.__name__}_{argval}"
                    y_pred_key = f"y_{scitype}_{argval}"

                # make prediction
                if y_pred_key not in y_preds_cache.keys():
                    start_pred = time.perf_counter()
                    y_pred = method(X_test, **pred_args)
                    pred_time = time.perf_counter() - start_pred
                    temp_result[time_key] = [pred_time]
                    y_preds_cache[y_pred_key] = [y_pred]
                else:
                    y_pred = y_preds_cache[y_pred_key][0]

                if scitype == "pred_proba":
                    if "pos_label" in inspect.signature(metric).parameters:
                        pos_label = 1
                        score = metric(y_test, y_pred[:, 1], pos_label=pos_label)
                    else:
                        score = metric(y_test, y_pred)
                else:
                    score = metric(y_test, y_pred)
                temp_result[result_key] = [score]

    except Exception as e:
        if error_score == "raise":
            raise e
        else:  # assign default value when fitting failed
            for scitype in scoring:
                temp_result[f"{scitype}_time"] = [pred_time]
                if return_data:
                    temp_result[f"y_{scitype}"] = [y_pred]
                for metric in scoring.get(scitype):
                    temp_result[f"test_{metric.__name__}"] = [score]
            warnings.warn(
                f"""
                In evaluate, fitting of {estimator_type}
                {type(estimator).__name__} failed,
                you can set error_score='raise' in evaluate to see
                the exception message.
                Fit failed for the {i}-th data split, on training data y_train
                , and len(y_train)={len(y_train)}.
                The score will be set to {error_score}.
                Failed {estimator_type} with parameters: {estimator}.
                """,
                FitFailedWarning,
                stacklevel=2,
            )

    # Storing the remaining evaluate detail
    temp_result["fit_time"] = [fit_time]

    if return_data:
        temp_result["X_train"] = [X_train]
        temp_result["X_test"] = [X_test]
        temp_result["y_train"] = [y_train]
        temp_result["y_test"] = [y_test]
        temp_result.update(y_preds_cache)
    result = pd.DataFrame(temp_result)
    column_order = _get_column_order_and_datatype(scoring, return_data)
    result = result.reindex(columns=column_order.keys())

    return result


def _run_evaluate(
    estimator,
    estimator_type,
    cv,
    scoring,
    X,
    y,
    return_data=False,
    error_score=np.nan,
    backend=None,
    backend_params=None,
):
    """Run cross-validated evaluation for a panel-data estimator.

    This is the shared body of ``classification.model_evaluation.evaluate``
    and ``regression.model_evaluation.evaluate``.

    Parameters
    ----------
    estimator : sktime estimator
        The estimator to evaluate.
    estimator_type : str
        Label for the estimator type, used in warning messages.
        E.g. ``"classifier"`` or ``"regressor"``.
    cv : int, cross-validation generator, or None
        Cross-validation splitting strategy.
    scoring : dict
        Validated scoring dict from ``_check_scores``.
        Keys are prediction types (e.g. ``"pred"``, ``"pred_proba"``),
        values are lists of metric callables.
    X : panel data container
        Panel features.
    y : tabular data container
        Target variable.
    return_data : bool, default=False
        Whether to return train/test/pred data in results.
    error_score : "raise" or numeric, default=np.nan
        Score to assign if fitting fails.
    backend : str or None, default=None
        Parallelization backend.
    backend_params : dict or None, default=None
        Additional backend parameters.

    Returns
    -------
    results : pd.DataFrame
        Cross-validation results.
    """
    if backend in ["dask", "dask_lazy"]:
        if not _check_soft_dependencies("dask", severity="none"):
            raise RuntimeError(
                "running evaluate with backend='dask' requires the dask package "
                "installed, but dask is not present in the python environment"
            )

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

    _evaluate_fold_kwargs = {
        "estimator": estimator,
        "estimator_type": estimator_type,
        "scoring": scoring,
        "return_data": return_data,
        "error_score": error_score,
    }

    def gen_y_X_train_test(y, X, cv):
        """Generate joint splits of y, X as per cv.

        Yields
        ------
        y_train : i-th train split of y as per cv
        y_test : i-th test split of y as per cv
        X_train : i-th train split of X as per cv.
        X_test : i-th test split of X as per cv.
        """
        instance_idx = X.index.get_level_values(0).unique()

        for train_instance_idx, test_instance_idx in cv.split(instance_idx):
            train_instances = instance_idx[train_instance_idx]
            test_instances = instance_idx[test_instance_idx]

            X_train = X.loc[X.index.get_level_values(0).isin(train_instances)]
            X_test = X.loc[X.index.get_level_values(0).isin(test_instances)]

            y_train = y.iloc[train_instance_idx]
            y_test = y.iloc[test_instance_idx]

            yield y_train, y_test, X_train, X_test

    # generator for y and X splits to iterate over below
    yx_splits = gen_y_X_train_test(y, X, _cv)

    results = parallelize(
        fun=_evaluate_fold,
        iter=enumerate(yx_splits),
        meta=_evaluate_fold_kwargs,
        backend=backend,
        backend_params=backend_params,
    )

    # final formatting of dask dataframes
    if backend in ["dask", "dask_lazy"]:
        import dask.dataframe as dd

        metadata = _get_column_order_and_datatype(scoring, return_data)

        results = dd.from_delayed(results, meta=metadata)
        if backend == "dask":
            results = results.compute()
    else:
        results = pd.concat(results)

    # final formatting of results DataFrame
    results = results.reset_index(drop=True)

    return results
