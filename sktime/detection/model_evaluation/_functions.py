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

from copy import deepcopy
import time
import warnings
import collections.abc

import numpy as np
import pandas as pd

from sktime.detection.base import BaseDetector
from sktime.exceptions import FitFailedWarning
from sktime.utils.adapters._safe_call import _method_has_arg
from sktime.performance_metrics.detection import WindowedF1Score

__all__ = ["evaluate"]


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

    return y.loc[mask_train].reset_index(drop=True), y.loc[mask_test].reset_index(drop=True)


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

    Parameters
    ----------
    detector : BaseDetector
    cv : splitter with `split_series` yielding (train_idx, test_idx) iloc arrays
    X : pd.Series or pd.DataFrame
    y : optional labels (points or segments) as DataFrame
    strategy : {"refit","update","no-update_params"}
    scoring : metric or list of metrics (default WindowedF1Score)

    Returns
    -------
    pd.DataFrame
        per-fold results with timing and metric columns.
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
            try:
                setattr(metric, "name", mname)
            except Exception:
                pass
        metric_names.append(str(mname))

    results = []
    fitted_det = None

    for i, (train_idx, test_idx) in enumerate(cv.split_series(X)):
        fit_time = np.nan
        pred_time = np.nan
        scores = {mn: error_score for mn in metric_names}
        y_pred = pd.DataFrame()
        y_train, y_test = _coerce_y_split(y, train_idx, test_idx, X=X)

        cutoff = pd.NA

        try:
            start_fit = time.perf_counter()
            if i == 0 or strategy == "refit":
                fitted_det = detector.clone()
                if _method_has_arg(fitted_det._fit, "y"):
                    fitted_det.fit(X=X.iloc[train_idx], y=y_train if y is not None else None)
                else:
                    fitted_det.fit(X=X.iloc[train_idx])
            else:
                # reuse fitted_det from previous fold
                update_params = True if strategy == "update" else False
                if _method_has_arg(fitted_det.update, "update_params"):
                    # include y if supported by public update
                    if _method_has_arg(fitted_det.update, "y"):
                        fitted_det.update(
                            X=X.iloc[train_idx],
                            y=y_train if y is not None else None,
                            update_params=update_params,
                        )
                    else:
                        fitted_det.update(X=X.iloc[train_idx], update_params=update_params)
                else:
                    # fallback to calling public update without update_params
                    if _method_has_arg(fitted_det.update, "y"):
                        fitted_det.update(X=X.iloc[train_idx], y=y_train if y is not None else None)
                    else:
                        fitted_det.update(X=X.iloc[train_idx])
            fit_time = time.perf_counter() - start_fit

            # predict
            start_pred = time.perf_counter()
            y_pred = fitted_det.predict(X.iloc[test_idx])
            pred_time = time.perf_counter() - start_pred

            # compute metrics individually
            for metric, mn in zip(scoring, metric_names):
                requires_y = True
                if hasattr(metric, "get_tag"):
                    try:
                        requires_y = metric.get_tag("requires_y_true", raise_error=False)
                    except Exception:
                        requires_y = True

                if requires_y:
                    scores[mn] = metric(y_test, y_pred, X.iloc[test_idx] if hasattr(X, "iloc") else X)
                else:
                    try:
                        scores[mn] = metric(y_pred, X.iloc[test_idx])
                    except TypeError:
                        scores[mn] = metric(y_pred)

            cutoff = pd.NA
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
                    'fitted_det' in locals()
                    and fitted_det is not None
                    and hasattr(fitted_det, "get_tag")
                    and fitted_det.get_tag("fit_is_empty", False)
                ):
                    suppress_warn = True
            except Exception:
                suppress_warn = False

            if not suppress_warn:
                warnings.warn(
                    f"In evaluate, fitting/predict of detector {type(detector).__name__} failed: {e}",
                    FitFailedWarning,
                )

        row = {
            "fit_time": fit_time,
            "pred_time": pred_time,
            "len_train_window": int(len(train_idx)),
            "cutoff": cutoff,
        }

        for mn in metric_names:
            row[f"test_{mn}"] = scores.get(mn, error_score)

        if return_data:
            row["y_train"] = y_train
            row["y_test"] = y_test
            row["y_pred"] = y_pred
        if return_model:
            try:
                row["fitted_detector"] = deepcopy(fitted_det)
            except Exception:
                row["fitted_detector"] = fitted_det

        results.append(row)

    result = pd.DataFrame(results)
    return result
