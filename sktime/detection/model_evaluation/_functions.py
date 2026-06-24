#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating detection models."""

__author__ = ["Nischal1425"]
__all__ = ["evaluate"]

import time
import warnings

import numpy as np
import pandas as pd

from sktime.exceptions import FitFailedWarning
from sktime.utils.parallel import parallelize


def _check_scoring(scoring):
    """Validate scoring and coerce to list of BaseDetectionMetric.

    Parameters
    ----------
    scoring : BaseDetectionMetric, list of BaseDetectionMetric, or None
        Scoring metric(s) for detection evaluation.
        If None, defaults to ``WindowedF1Score()``.

    Returns
    -------
    scoring : list of BaseDetectionMetric
        Validated list of scoring metrics.

    Raises
    ------
    TypeError
        If any scorer is not a ``BaseDetectionMetric`` instance.
    """
    if scoring is None:
        from sktime.performance_metrics.detection import WindowedF1Score

        return [WindowedF1Score()]

    if not isinstance(scoring, list):
        scoring = [scoring]

    from sktime.performance_metrics.detection._base import BaseDetectionMetric

    for metric in scoring:
        if not isinstance(metric, BaseDetectionMetric):
            raise TypeError(
                f"All scorers must be instances of BaseDetectionMetric, "
                f"but found {type(metric)}. "
                f"Use metrics from sktime.performance_metrics.detection."
            )
    return scoring


def _check_cv(cv):
    """Validate and coerce cv parameter.

    Parameters
    ----------
    cv : int, or sklearn splitter object, or None
        Cross-validation strategy.

        * If ``None`` or ``int``, uses ``KFold`` with ``shuffle=False``.
          If ``int``, sets ``n_splits`` to the given value.
          If ``None``, defaults to ``n_splits=3``.
        * If sklearn splitter, used directly.

    Returns
    -------
    cv : sklearn splitter object
        Validated cross-validation splitter.
    """
    if cv is None:
        from sklearn.model_selection import KFold

        return KFold(n_splits=3, shuffle=False)

    if isinstance(cv, int):
        from sklearn.model_selection import KFold

        return KFold(n_splits=cv, shuffle=False)

    return cv


def _filter_events_to_range(y, idx_range):
    """Filter detection events to a given index range and re-index ilocs.

    Detection ground truth ``y`` is a ``pd.DataFrame`` with an ``ilocs``
    column containing iloc-based indices into the full time series ``X``.
    When ``X`` is split into train/test, the ground truth events must be
    filtered to only include events within the split, and their ``ilocs``
    must be re-indexed to be relative to the start of the sub-series.

    Parameters
    ----------
    y : pd.DataFrame with ``ilocs`` column, or None
        Detection events in sparse format.
    idx_range : array-like of int
        The iloc indices (into the original X) that define this split.

    Returns
    -------
    y_filtered : pd.DataFrame with ``ilocs`` column, or None
        Filtered and re-indexed events, or ``None`` if input is ``None``.
    """
    if y is None:
        return None

    if len(y) == 0:
        return y.copy()

    idx_set = set(idx_range)
    start = idx_range[0] if len(idx_range) > 0 else 0

    mask = y["ilocs"].apply(lambda x: x in idx_set)
    y_filtered = y.loc[mask].copy()

    if len(y_filtered) > 0:
        y_filtered["ilocs"] = y_filtered["ilocs"].apply(lambda x: x - start)

    y_filtered = y_filtered.reset_index(drop=True)
    return y_filtered


def _evaluate_fold(x, meta):
    """Evaluate a single CV fold for detection.

    This function is designed to be called via ``parallelize``.

    Parameters
    ----------
    x : tuple of (int, (train_indices, test_indices))
        Fold index and train/test index arrays.
    meta : dict
        Dictionary containing shared evaluation metadata:
        - detector : BaseDetector instance to evaluate
        - scoring : list of BaseDetectionMetric
        - return_data : bool
        - error_score : "raise" or numeric
        - X : pd.DataFrame or pd.Series, full time series
        - y : pd.DataFrame or None, full ground truth events

    Returns
    -------
    result : pd.DataFrame
        Single-row DataFrame with scores, timing, and optionally data.
    """
    i, (train_idx, test_idx) = x

    detector = meta["detector"]
    scoring = meta["scoring"]
    return_data = meta["return_data"]
    error_score = meta["error_score"]
    X = meta["X"]
    y = meta["y"]

    # split X into train/test
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    # split y (ground truth events) into train/test
    # if y is None (unsupervised), use empty DataFrames with ilocs column
    # so detection metrics can evaluate without TypeError
    _empty_y = pd.DataFrame({"ilocs": pd.Series(dtype="int64")})
    y_train = _filter_events_to_range(y, train_idx) if y is not None else _empty_y
    y_test = _filter_events_to_range(y, test_idx) if y is not None else _empty_y

    # default values in case of failure
    fit_time = np.nan
    pred_time = np.nan
    result = {}

    try:
        # fit
        start_fit = time.perf_counter()
        detector_clone = detector.clone()
        detector_clone.fit(X=X_train, y=y_train)
        fit_time = time.perf_counter() - start_fit

        # predict
        start_pred = time.perf_counter()
        y_pred = detector_clone.predict(X=X_test)
        pred_time = time.perf_counter() - start_pred

        # score
        for metric in scoring:
            metric_name = metric.__class__.__name__
            score = metric.evaluate(y_true=y_test, y_pred=y_pred, X=X_test)
            result[f"test_{metric_name}"] = score

    except Exception as e:
        if error_score == "raise":
            raise e
        else:
            warnings.warn(
                f"Detector fit failed with error: {e}. Setting score to {error_score}.",
                FitFailedWarning,
                stacklevel=2,
            )
            for metric in scoring:
                metric_name = metric.__class__.__name__
                result[f"test_{metric_name}"] = error_score
            y_pred = None

    result["fit_time"] = fit_time
    result["pred_time"] = pred_time
    result["len_train_window"] = len(train_idx)

    if return_data:
        result["y_train"] = [y_train]
        result["y_test"] = [y_test]
        result["y_pred"] = [y_pred]
        result["X_train"] = [X_train]
        result["X_test"] = [X_test]

    return pd.DataFrame(result, index=[i])


def evaluate(
    detector,
    X,
    y=None,
    cv=None,
    scoring=None,
    return_data=False,
    error_score=np.nan,
    backend=None,
    backend_params=None,
):
    """Evaluate a detection estimator using cross-validation.

    Evaluates a detector by fitting it on training splits and scoring
    predictions on test splits. Supports temporal and non-temporal
    cross-validation strategies.

    Parameters
    ----------
    detector : BaseDetector
        The detection estimator to evaluate. Must implement ``fit(X, y)``
        and ``predict(X)``.

    X : pd.DataFrame or pd.Series
        Time series data to evaluate the detector on.

    y : pd.DataFrame with ``ilocs`` column, optional (default=None)
        Ground truth events in sparse format. Each row represents a known
        event, with the ``ilocs`` column containing the iloc index into ``X``
        where the event occurs.

        Required for supervised detectors and supervised metrics.
        If ``None``, the detector is fit without labels and metrics
        that require ``y_true`` will receive ``None``.

    cv : int, sklearn splitter, or None (default=None)
        Cross-validation strategy.

        * If ``None``, uses ``KFold(n_splits=3, shuffle=False)``.
        * If ``int``, uses ``KFold(n_splits=cv, shuffle=False)``.
        * If sklearn splitter object (e.g., ``KFold``, ``TimeSeriesSplit``),
          used directly. The splitter operates on the time index of ``X``.

        ``shuffle=False`` is the default because detection tasks are
        inherently temporal — shuffling would break temporal dependencies
        that detectors (especially change point detectors) rely on.

    scoring : BaseDetectionMetric, list of BaseDetectionMetric, or None
        Scoring metric(s) for evaluation.

        * If ``None``, defaults to ``WindowedF1Score()``.
        * Each metric must be an instance of ``BaseDetectionMetric``.

        Available metrics include ``WindowedF1Score``, ``DirectedChamfer``,
        ``DirectedHausdorff``, ``RandIndex``, ``DetectionCount``,
        ``TimeSeriesAUPRC``.

    return_data : bool, optional (default=False)
        Whether to include train/test data and predictions in the results.
        If ``True``, the returned DataFrame will include columns
        ``y_train``, ``y_test``, ``y_pred``, ``X_train``, ``X_test``.

    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator
        fitting. If set to ``"raise"``, the exception is raised. If a
        numeric value is given, ``FitFailedWarning`` is raised.

    backend : str, optional (default=None)
        Parallelization backend. See ``sktime.utils.parallel.parallelize``
        for details. Options include ``"loky"``, ``"multiprocessing"``,
        ``"threading"``, ``"dask"``, ``"ray"``.

    backend_params : dict, optional (default=None)
        Additional parameters for the parallelization backend.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with one row per CV fold. Columns include:

        * ``test_<MetricName>`` : float — score for each metric
        * ``fit_time`` : float — time in seconds to fit the detector
        * ``pred_time`` : float — time in seconds to predict
        * ``len_train_window`` : int — number of training samples

        If ``return_data=True``, additional columns:

        * ``y_train`` : ground truth events for the training split
        * ``y_test`` : ground truth events for the test split
        * ``y_pred`` : predicted events on the test split
        * ``X_train`` : training time series
        * ``X_test`` : test time series

    See Also
    --------
    sktime.benchmarking.detection.DetectionBenchmark :
        High-level benchmarking interface for detection estimators.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.model_selection import KFold
    >>> from sktime.detection.dummy import DummyRegularAnomalies
    >>> from sktime.detection.model_evaluation import evaluate
    >>> from sktime.performance_metrics.detection import WindowedF1Score
    >>> X = pd.DataFrame({"value": range(50)})
    >>> detector = DummyRegularAnomalies(step_size=5)
    >>> results = evaluate(
    ...     detector=detector,
    ...     X=X,
    ...     cv=KFold(n_splits=3, shuffle=False),
    ...     scoring=WindowedF1Score(margin=2),
    ... )
    """
    from sktime.detection.base import BaseDetector

    if not isinstance(detector, BaseDetector):
        raise TypeError(
            f"detector must be a BaseDetector instance, but found {type(detector)}."
        )

    scoring = _check_scoring(scoring)
    cv = _check_cv(cv)

    # convert X to DataFrame if Series
    if isinstance(X, pd.Series):
        X = X.to_frame()

    # generate CV splits on the time index
    n_samples = len(X)
    indices = np.arange(n_samples)
    splits = list(cv.split(indices))

    meta = {
        "detector": detector,
        "scoring": scoring,
        "return_data": return_data,
        "error_score": error_score,
        "X": X,
        "y": y,
    }

    # build iterable of (fold_index, (train, test))
    fold_inputs = list(enumerate(splits))

    results = parallelize(
        fun=_evaluate_fold,
        iter=fold_inputs,
        meta=meta,
        backend=backend,
        backend_params=backend_params,
    )

    results_df = pd.concat(results, axis=0).reset_index(drop=True)

    return results_df
