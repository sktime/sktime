"""
Evaluation utilities for detectors.

Light-weight evaluate function for detectors. Mirrors the forecaster
`evaluate` utility in spirit but is simplified for detection tasks.

Features:
- Supports point and segment encodings for `y` (columns `ilocs` or `start`/`end`).
- Default metric: :class:`~sktime.performance_metrics.detection.WindowedF1Score`.
- Supports strategies: "refit", "update", "no-update_params" (behavior similar
  to forecasting.evaluate: refit resets the detector each fold, update reuses
  and calls update on it).

Notes / assumptions:
- Splitting of labels to train/test is done in iloc units (cv.split_series is
  expected to return iloc indices). For segments, a segment is considered part
  of the training set if its end iloc <= last train iloc, and part of the test
  set if it overlaps the test iloc interval.
"""

import collections.abc
import logging
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.exceptions import FitFailedWarning
from sktime.performance_metrics.detection import WindowedF1Score
from sktime.utils.adapters._safe_call import _method_has_arg
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.parallel import parallelize

__all__ = ["evaluate"]


def _get_column_order_and_datatype(
    metric_names: list, return_data: bool = False, return_model: bool = False
) -> dict:
    """
    Get ordered column names and simple datatypes for results metadata.

    Parameters
    ----------
    metric_names : list
        List of metric names (strings) as prepared in evaluate.
    return_data : bool, optional
        Whether y_train/y_test/y_pred columns will be returned.
    return_model : bool, optional
        Whether fitted_detector column will be returned.

    Returns
    -------
    dict
        Ordered mapping column_name -> dtype string for use as meta in dask.
    """
    fit_metadata = {
        "fit_time": "float",
        "pred_time": "float",
        "len_train_window": "int",
        "cutoff": "object",
    }
    metrics_metadata = {}
    for mn in metric_names:
        metrics_metadata[f"test_{mn}"] = "float"

    if return_data:
        fit_metadata.update(
            {"y_train": "object", "y_test": "object", "y_pred": "object"}
        )
    if return_model:
        fit_metadata.update({"fitted_detector": "object"})

    # final ordering: metric columns first, then fit metadata
    metrics_metadata.update(fit_metadata)
    return metrics_metadata.copy()


def _coerce_y_split(y, train_idx, test_idx, X=None):
    """
    Split `y` into (y_train, y_test) according to iloc-based train/test idx.

    Supports the following `y` encodings:
    - pd.DataFrame with column `ilocs` where each row is an int (point) or
      pd.Interval (segment), or iterable (start, end)
    - pd.DataFrame with `start` and `end` columns representing segments

    Returns two DataFrames (y_train, y_test) with the same columns as `y`.
    If `y` is None, returns (None, None).
    """
    if y is None:
        return None, None

    y = pd.DataFrame(y).reset_index(drop=True)

    has_ilocs = "ilocs" in y.columns
    has_start_end = ("start" in y.columns) and ("end" in y.columns)
    if not (has_ilocs or has_start_end):
        return pd.DataFrame(columns=y.columns), pd.DataFrame(columns=y.columns)

    train_max = int(np.max(train_idx)) if len(train_idx) > 0 else -1
    test_min = int(np.min(test_idx)) if len(test_idx) > 0 else None
    test_max = int(np.max(test_idx)) if len(test_idx) > 0 else None
    test_set = set(np.asarray(test_idx).astype(int).tolist())

    starts = []
    ends = []

    if has_ilocs:
        raw_ilocs = list(y["ilocs"])
        ilocs_mapped = []
        need_map = False
        for v in raw_ilocs:
            if pd.isna(v):
                ilocs_mapped.append(np.nan)
            elif isinstance(v, (int, np.integer)):
                ilocs_mapped.append(int(v))
            else:
                need_map = True
                ilocs_mapped.append(v)

        if need_map and X is not None:
            try:
                # use pandas Index.get_indexer to map loc values to iloc positions
                idx = X.index
                mapped = idx.get_indexer(pd.Index(ilocs_mapped))
                # unmapped positions are -1 -> convert to NaN
                for m in mapped:
                    if m >= 0:
                        starts.append(int(m))
                        ends.append(int(m))
                    else:
                        starts.append(np.nan)
                        ends.append(np.nan)
            except Exception:
                for _ in ilocs_mapped:
                    starts.append(np.nan)
                    ends.append(np.nan)
        else:
            for v in ilocs_mapped:
                if pd.isna(v):
                    starts.append(np.nan)
                    ends.append(np.nan)
                elif isinstance(v, (int, np.integer)):
                    starts.append(int(v))
                    ends.append(int(v))
                elif isinstance(v, pd.Interval):
                    starts.append(int(v.left))
                    ends.append(int(v.right))
                else:
                    try:
                        s, e = list(v)
                        starts.append(int(s) if s is not None else np.nan)
                        ends.append(int(e) if e is not None else np.nan)
                    except Exception:
                        starts.append(np.nan)
                        ends.append(np.nan)
    else:
        for s, e in zip(y["start"], y["end"]):
            starts.append(int(s) if not pd.isna(s) else np.nan)
            ends.append(int(e) if not pd.isna(e) else np.nan)

    starts = np.array(starts, dtype=float)
    ends = np.array(ends, dtype=float)

    mask_train = np.zeros(len(y), dtype=bool)
    mask_test = np.zeros(len(y), dtype=bool)

    for i in range(len(y)):
        if pd.isna(starts[i]) and pd.isna(ends[i]):
            continue
        s = int(starts[i])
        e = int(ends[i])
        if s == e:
            if s in test_set:
                mask_test[i] = True
            if s <= train_max:
                mask_train[i] = True
        else:
            # segment -> training membership if entirely before or at train_max
            if e <= train_max:
                mask_train[i] = True
            if (test_min is not None) and (test_max is not None):
                if not (e < test_min or s > test_max):
                    mask_test[i] = True

    return y.loc[mask_train].reset_index(drop=True), y.loc[mask_test].reset_index(
        drop=True
    )


def evaluate(
    detector,
    cv,
    X,
    y=None,
    strategy: str = "refit",
    scoring: collections.abc.Callable | list[collections.abc.Callable] | None = None,
    return_data: bool = False,
    error_score: str | int | float = np.nan,
    backend: str | None = None,
    backend_params: dict | None = None,
    return_model: bool = False,
):
    """
    Evaluate a detector across folds produced by `cv`.

    Lightweight utility to run temporal cross-validation for detectors and
    collect per-fold runtime and metric results.

    Parameters
    ----------
    detector : BaseDetector
        Detector instance to evaluate. Must inherit from :class:`BaseDetector`.

    cv : splitter
        Splitter with ``split_series`` method yielding ``(train_idx, test_idx)``
        iloc-based index arrays for each fold.

    X : pd.Series or pd.DataFrame
        Input series/dataframe containing features or the series being
        evaluated. Used to map label locations to iloc positions when needed.

    y : pd.DataFrame or None, optional (default=None)
        Labels for detection tasks. Supported encodings are a DataFrame with
        an ``ilocs`` column (points or pd.Interval for segments) or ``start``
        and ``end`` columns for segments. If ``None``, metrics that require
        ground truth will be skipped or will receive ``None`` depending on the
        metric implementation.

    strategy : {"refit", "update", "no-update_params"}, optional
        How the detector ingests new data between folds. Default = ``"refit"``.

        * ``"refit"`` : clone and fit a fresh detector for each fold.
        * ``"update"`` : reuse the fitted detector and call its ``update``
          method with new training data (if supported).
        * ``"no-update_params"`` : call ``update`` with ``update_params=False``
          where supported (behaves like reusing parameters without refitting).

    scoring : callable or list of callables, optional
        Metric or list of metrics to compute on each fold. Each metric should
        accept arguments in the form ``metric(y_true, y_pred, X)`` or
        ``metric(y_pred, X)`` depending on whether the metric requires true
        labels. If ``None``, defaults to
        :class:`~sktime.performance_metrics.detection.WindowedF1Score`.

    return_data : bool, optional (default=False)
        If ``True``, include columns ``y_train``, ``y_test`` and ``y_pred`` in
        the returned DataFrame for each fold.

    error_score : {"raise"} or numeric, optional (default=np.nan)
        Value assigned to the metric if fitting or prediction raises an
        exception for a fold. If set to ``"raise"``, the exception is raised
        instead of being captured.

    backend : str or None, optional (default=None)
        Parallelization backend to use when running evaluation. If ``None``
        execution is sequential. Supported backends in other evaluate
        utilities include ``"loky"``, ``"multiprocessing"``, ``"threading"``,
        ``"dask"``, etc. Note: in the detection evaluate implementation only
        simple sequential execution is used; this parameter is accepted for
        API parity and future extension.

    backend_params : dict or None, optional
        Additional backend configuration passed through to the parallel
        execution utilities when ``backend`` is not ``None``. Ignored when
        ``backend`` is ``None``.

    return_model : bool, optional (default=False)
        If ``True``, include a ``fitted_detector`` column in the returned
        DataFrame containing (a deepcopy of) the fitted detector for each fold
        when possible.

    Returns
    -------
    pd.DataFrame
        Per-fold results. Columns include:

        - ``test_{metric.name}`` : (float) metric value for each scorer provided
        - ``fit_time`` : (float) time in seconds spent fitting or updating
        - ``pred_time`` : (float) time in seconds spent predicting
        - ``len_train_window`` : (int) length of the train window (number of iloc)
        - ``cutoff`` : (int) last train iloc used as cutoff for the fold

        If ``return_data=True``, the DataFrame will also contain ``y_train``,
        ``y_test`` and ``y_pred`` columns. If ``return_model=True``, the
        DataFrame will contain a ``fitted_detector`` column with the fitted
        detector per fold (deepcopied when possible).
    """
    if not isinstance(detector, BaseDetector):
        raise TypeError("`detector` must inherit from BaseDetector")

    if scoring is None:
        scoring = WindowedF1Score()

    if not isinstance(scoring, list):
        scoring = [scoring]

    # prepare metric names
    metric_names = []
    for metric in scoring:
        mname = getattr(metric, "name", None)
        if mname is None:
            cls = getattr(metric, "__class__", None)
            mname = cls.__name__ if isinstance(cls, type) else str(cls)
            if hasattr(metric, "__dict__"):
                try:
                    setattr(metric, "name", mname)
                except Exception:
                    logging.getLogger(__name__).debug(
                        "failed to set metric.name", exc_info=True
                    )
        metric_names.append(str(mname))

    # helper to validate strategy value
    def _check_strategy(strategy_val):
        valid = ("refit", "update", "no-update_params")
        if strategy_val not in valid:
            raise ValueError(f"strategy must be one of {valid}, got {strategy_val}")

    _check_strategy(strategy)

    # prepare meta kwargs for per-window evaluation
    _evaluate_window_kwargs = {
        "detector": detector,
        "X": X,
        "y": y,
        "scoring": scoring,
        "metric_names": metric_names,
        "strategy": strategy,
        "return_data": return_data,
        "return_model": return_model,
        "error_score": error_score,
    }

    # backend soft-dependency checks
    if backend in ["dask", "dask_lazy"]:
        if not _check_soft_dependencies("dask", severity="none"):
            raise RuntimeError(
                "running evaluate with backend='dask' requires the dask package "
                "installed, but dask is not present in the python environment"
            )

    if backend == "ray" and not _check_soft_dependencies("ray", severity="none"):
        raise RuntimeError(
            "running evaluate with backend='ray' requires the ray package "
            "installed, but ray is not present in the python environment"
        )

    # sequential strategies cannot be parallelized
    not_parallel = strategy in ["update", "no-update_params"]

    # iterate over folds / dispatch by backend
    results = []
    res = None
    if not_parallel:
        fitted_det = None
        for x in enumerate(cv.split_series(X)):
            # pass along current fitted detector when applicable
            if fitted_det is not None:
                _evaluate_window_kwargs["fitted_detector"] = fitted_det

            row_res = _evaluate_window(x, _evaluate_window_kwargs)
            if isinstance(row_res, tuple):
                row, fitted_det = row_res
            else:
                row = row_res
                fitted_det = None
            results.append(row)
        # results is a list of pd.DataFrame
    else:
        # parallel / refit mode
        backend_in = "dask_lazy" if backend == "dask" else backend
        res = parallelize(
            fun=_evaluate_window,
            iter=enumerate(cv.split_series(X)),
            meta=_evaluate_window_kwargs,
            backend=backend_in,
            backend_params=backend_params,
        )
        # res is either a list of pd.DataFrame (sequential joblib) or a list of
        # dask delayed objects (when backend_in == 'dask_lazy'). We'll handle
        # dask specially below.
        # if backend_in is not dask_lazy, parallelize returned concrete results
        # which we should collect into `results` for later concatenation.
        if backend_in != "dask_lazy":
            results = res

    # final formatting / aggregation
    if backend in ["dask", "dask_lazy"] and not not_parallel:
        # import dask lazily to avoid hard dependency at module import time
        try:
            import importlib

            dd = importlib.import_module("dask.dataframe")
        except Exception:
            raise RuntimeError(
                "running evaluate with backend='dask' requires the dask package "
                "installed, but dask is not present in the python environment"
            )

        metadata = _get_column_order_and_datatype(
            metric_names, return_data, return_model=return_model
        )

        results = dd.from_delayed(res, meta=metadata)
        if backend == "dask":
            results = results.compute()
    else:
        # results is a list of pd.DataFrame collected above (sequential or joblib)
        results = pd.concat(results)

    results = results.reset_index(drop=True)
    return results


def _evaluate_window(x, meta):
    """Evaluate a single fold for detectors.

    Parameters
    ----------
    x : tuple
        Element from ``enumerate(cv.split_series(X))``; shape ``(i, (train_idx,
        test_idx))``.
    meta : dict
        Dictionary with keys: detector, X, y, scoring, metric_names, strategy,
        return_data, return_model, error_score

    Returns
    -------
    dict or (dict, BaseDetector)
        Row dict with results for the fold. If strategy requires returning the
        fitted detector (sequential update mode), returns a tuple
        (row_dict, fitted_detector).
    """
    i, (train_idx, test_idx) = x
    detector = meta["detector"]
    X = meta["X"]
    y = meta.get("y", None)
    scoring = meta["scoring"]
    metric_names = meta["metric_names"]
    strategy = meta["strategy"]
    return_data = meta["return_data"]
    return_model = meta["return_model"]
    error_score = meta["error_score"]

    fit_time = np.nan
    pred_time = np.nan
    scores = dict.fromkeys(metric_names, error_score)
    y_pred = pd.DataFrame()
    y_train, y_test = _coerce_y_split(y, train_idx, test_idx, X=X)

    cutoff = pd.NA

    fitted_det = None

    try:
        start_fit = time.perf_counter()
        provided_fitted = meta.get("fitted_detector", None)
        if provided_fitted is not None and strategy != "refit":
            fitted_det = provided_fitted
            update_params = True if strategy == "update" else False
            if _method_has_arg(fitted_det.update, "update_params"):
                if _method_has_arg(fitted_det.update, "y"):
                    fitted_det.update(
                        X=X.iloc[train_idx],
                        y=y_train if y is not None else None,
                        update_params=update_params,
                    )
                else:
                    fitted_det.update(X=X.iloc[train_idx], update_params=update_params)
            else:
                if _method_has_arg(fitted_det.update, "y"):
                    fitted_det.update(
                        X=X.iloc[train_idx], y=y_train if y is not None else None
                    )
                else:
                    fitted_det.update(X=X.iloc[train_idx])
        else:
            fitted_det = detector.clone()
            if _method_has_arg(fitted_det._fit, "y"):
                fitted_det.fit(
                    X=X.iloc[train_idx], y=y_train if y is not None else None
                )
            else:
                fitted_det.fit(X=X.iloc[train_idx])
        fit_time = time.perf_counter() - start_fit

        # predict
        start_pred = time.perf_counter()
        y_pred = fitted_det.predict(X.iloc[test_idx])
        pred_time = time.perf_counter() - start_pred

        # compute metrics
        for metric, mn in zip(scoring, metric_names):
            requires_y = True
            if hasattr(metric, "get_tag"):
                try:
                    requires_y = metric.get_tag("requires_y_true", raise_error=False)
                except Exception:
                    requires_y = True

            if requires_y:
                scores[mn] = metric(
                    y_test, y_pred, X.iloc[test_idx] if hasattr(X, "iloc") else X
                )
            else:
                try:
                    scores[mn] = metric(y_pred, X.iloc[test_idx])
                except TypeError:
                    scores[mn] = metric(y_pred)

        try:
            if len(train_idx) > 0:
                cutoff = int(np.asarray(train_idx)[-1])
        except Exception:
            cutoff = pd.NA

    except Exception as e:
        if error_score == "raise":
            raise
        suppress_warn = False
        try:
            if hasattr(detector, "get_tag") and detector.get_tag("fit_is_empty", False):
                suppress_warn = True
            elif (
                "fitted_det" in locals()
                and fitted_det is not None
                and hasattr(fitted_det, "get_tag")
                and fitted_det.get_tag("fit_is_empty", False)
            ):
                suppress_warn = True
        except Exception:
            suppress_warn = False

        if not suppress_warn:
            msg = (
                "In evaluate, fitting/predict of detector "
                f"{type(detector).__name__} failed: {e}"
            )
            warnings.warn(msg, FitFailedWarning)

    temp_result = {}
    # store metric values and timing as single-element lists to create DataFrame
    temp_result["fit_time"] = [fit_time]
    temp_result["pred_time"] = [pred_time]
    temp_result["len_train_window"] = [int(len(train_idx))]
    temp_result["cutoff"] = [cutoff]

    for mn in metric_names:
        temp_result[f"test_{mn}"] = [scores.get(mn, error_score)]

    if return_data:
        temp_result["y_train"] = [y_train]
        temp_result["y_test"] = [y_test]
        temp_result["y_pred"] = [y_pred]
    if return_model:
        try:
            temp_result["fitted_detector"] = [deepcopy(fitted_det)]
        except Exception:
            temp_result["fitted_detector"] = [fitted_det]

    result = pd.DataFrame(temp_result)
    # reorder columns according to metadata helper
    col_meta = _get_column_order_and_datatype(
        metric_names, return_data=return_data, return_model=return_model
    )
    # ensure result has all columns in the order of col_meta
    cols = [c for c in col_meta.keys() if c in result.columns]
    result = result.reindex(columns=cols)

    # return DataFrame in refit/parallel mode; sequential update returns
    # (DataFrame, fitted_det)
    return result if strategy == "refit" else (result, fitted_det)
