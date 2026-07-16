#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utilities for performance evaluation of time series clustering models."""

__author__ = ["Nischal1425"]
__all__ = ["evaluate"]

import collections.abc
import inspect
import time
import warnings

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datatypes import check_is_scitype, convert, convert_to
from sktime.exceptions import FitFailedWarning
from sktime.utils.parallel import parallelize

PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


def _classify_metric(metric):
    """Classify a clustering metric as internal or external.

    Internal metrics evaluate clustering quality using only the data and
    cluster assignments (e.g., ``silhouette_score(X, labels)``).

    External metrics compare cluster assignments against ground truth labels
    (e.g., ``adjusted_rand_score(labels_true, labels_pred)``).

    Parameters
    ----------
    metric : callable
        A sklearn-compatible clustering metric function.

    Returns
    -------
    str
        ``"internal"`` if the metric's first parameter is named ``X``,
        ``"external"`` otherwise.
    """
    if not callable(metric):
        raise ValueError(f"Metric {metric} is not callable")

    sig = inspect.signature(metric)
    params = list(sig.parameters.keys())

    # Internal metrics like silhouette_score, calinski_harabasz_score,
    # davies_bouldin_score have 'X' as first parameter.
    # External metrics like adjusted_rand_score, normalized_mutual_info_score
    # have 'labels_true' as first parameter.
    if len(params) >= 1 and params[0] == "X":
        return "internal"
    return "external"


def _check_scores(metrics):
    """Validate sklearn clustering metrics and categorize by type.

    Categorizes metrics as ``"internal"`` (requiring raw data X and cluster
    labels) or ``"external"`` (requiring ground truth and predicted labels).

    Parameters
    ----------
    metrics : sklearn metric function, list of sklearn metric functions, or None
        The clustering metrics to validate and categorize.

    Returns
    -------
    metrics_type : dict
        Dictionary where keys are metric types (``"internal"`` or
        ``"external"``) and values are lists of corresponding metrics.
    """
    if metrics is None:
        from sklearn.metrics import silhouette_score

        metrics = [silhouette_score]

    if not isinstance(metrics, list):
        metrics = [metrics]

    metrics_type = {}

    for metric in metrics:
        if not callable(metric):
            raise ValueError(f"Metric {metric} is not callable")

        scitype = _classify_metric(metric)

        if scitype not in metrics_type:
            metrics_type[scitype] = [metric]
        else:
            metrics_type[scitype].append(metric)

    return metrics_type


def _get_column_order_and_datatype(metric_types, return_data=True):
    """Get the ordered column name and input datatype of results."""
    y_metadata = {
        "X_train": "object",
        "X_test": "object",
        "y_train": "object",
        "y_test": "object",
    }
    fit_metadata = {"fit_time": "float"}
    metrics_metadata = {}

    for scitype in metric_types:
        time_key = f"{scitype}_time"
        fit_metadata[time_key] = "float"
        for metric in metric_types.get(scitype):
            result_key = f"test_{metric.__name__}"
            metrics_metadata[result_key] = "float"

        if return_data:
            y_metadata[f"y_{scitype}"] = "object"

    if return_data:
        fit_metadata.update(y_metadata)
    metrics_metadata.update(fit_metadata)
    return metrics_metadata.copy()


def _flatten_panel_to_2d(X_panel):
    """Convert panel data to 2D numpy array for sklearn clustering metrics.

    Internal clustering metrics like ``silhouette_score`` require a 2D
    feature matrix ``(n_samples, n_features)``. This function converts
    sktime panel data to this format by reshaping numpy3D
    ``(n_instances, n_dims, series_length)`` to
    ``(n_instances, n_dims * series_length)``.

    Parameters
    ----------
    X_panel : pd.DataFrame
        Panel data in a pandas mtype (e.g., pd-multiindex).

    Returns
    -------
    np.ndarray
        2D array of shape ``(n_instances, n_features)`` where
        ``n_features = n_dimensions * series_length``.
    """
    X_np = convert_to(X_panel, to_type="numpy3D", as_scitype="Panel")
    return X_np.reshape(X_np.shape[0], -1)


def _evaluate_fold(x, meta):
    """Evaluate a single CV fold for clustering.

    Parameters
    ----------
    x : tuple
        Tuple of ``(fold_index, (X_train, X_test, y_train, y_test))``
        where y_train/y_test may be None for unsupervised clustering.
    meta : dict
        Metadata dict with keys ``"clusterer"``, ``"scoring"``,
        ``"return_data"``, ``"error_score"``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with scores, timing, and optionally data.
    """
    i, (X_train, X_test, y_train, y_test) = x
    clusterer = meta["clusterer"]
    scoring = meta["scoring"]
    return_data = meta["return_data"]
    error_score = meta["error_score"]

    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    y_pred = pd.NA
    temp_result = {}
    y_preds_cache = {}

    try:
        # fit
        start_fit = time.perf_counter()

        clusterer = clusterer.clone()
        clusterer.fit(X=X_train)

        fit_time = time.perf_counter() - start_fit

        # predict
        for scitype in scoring:
            if scitype not in y_preds_cache:
                start_pred = time.perf_counter()
                y_pred = clusterer.predict(X_test)
                pred_time = time.perf_counter() - start_pred
                temp_result[f"{scitype}_time"] = [pred_time]
                y_preds_cache[scitype] = y_pred
            else:
                y_pred = y_preds_cache[scitype]

            for metric in scoring.get(scitype):
                result_key = f"test_{metric.__name__}"

                if scitype == "internal":
                    # Internal metrics: metric(X, labels)
                    X_test_flat = _flatten_panel_to_2d(X_test)

                    # silhouette_score requires at least 2 unique labels
                    n_unique = len(np.unique(y_pred))
                    if n_unique < 2:
                        score = np.nan
                    else:
                        score = metric(X_test_flat, y_pred)
                else:
                    # External metrics: metric(y_true, y_pred)
                    if y_test is None:
                        raise ValueError(
                            f"External metric '{metric.__name__}' requires "
                            f"ground truth labels (y), but y=None was provided. "
                            f"Use internal metrics (e.g., silhouette_score) for "
                            f"unsupervised clustering evaluation, or provide y."
                        )
                    # Flatten y_test to 1D — sklearn metrics expect 1D arrays
                    y_test_1d = y_test.values.ravel() if hasattr(y_test, "values") else np.asarray(y_test).ravel()
                    score = metric(y_test_1d, y_pred)
                temp_result[result_key] = [score]

    except Exception as e:
        if error_score == "raise":
            raise e
        else:
            for scitype in scoring:
                temp_result[f"{scitype}_time"] = [pred_time]
                if return_data:
                    temp_result[f"y_{scitype}"] = [y_pred]
                for metric in scoring.get(scitype):
                    temp_result[f"test_{metric.__name__}"] = [score]
            warnings.warn(
                f"""
                In evaluate, fitting of clusterer {type(clusterer).__name__} failed,
                you can set error_score='raise' in evaluate to see
                the exception message.
                Fit failed for the {i}-th data split, on training data X_train
                , and len(X_train)={len(X_train)}.
                The score will be set to {error_score}.
                Failed clusterer with parameters: {clusterer}.
                """,
                FitFailedWarning,
                stacklevel=2,
            )

    # Storing the remaining evaluate details
    temp_result["fit_time"] = [fit_time]

    if return_data:
        temp_result["X_train"] = [X_train]
        temp_result["X_test"] = [X_test]
        temp_result["y_train"] = [y_train]
        temp_result["y_test"] = [y_test]
        for scitype in y_preds_cache:
            temp_result[f"y_{scitype}"] = [y_preds_cache[scitype]]

    result = pd.DataFrame(temp_result)
    column_order = _get_column_order_and_datatype(scoring, return_data)
    result = result.reindex(columns=column_order.keys())

    return result


def evaluate(
    clusterer,
    cv=None,
    X=None,
    y=None,
    scoring: collections.abc.Callable | list[collections.abc.Callable] | None = None,
    return_data: bool = False,
    error_score: str | int | float = np.nan,
    backend: str | None = None,
    backend_params: dict | None = None,
):
    r"""Evaluate clusterer using cross-validation.

    All-in-one statistical performance benchmarking utility for clusterers,
    which runs a simple cross-validation experiment and returns a summary
    DataFrame.

    The experiment runs the following:

    1. Train and test folds are generated by ``cv.split(X)``.
    2. For each fold ``i`` in ``K`` folds:

        a. Fit the ``clusterer`` on the training set ``X_train``
        b. Predict cluster labels ``y_pred`` on the test set ``X_test``
        c. Compute the score via ``scoring(X_test, y_pred)`` for internal
           metrics or ``scoring(y_test, y_pred)`` for external metrics
        d. If ``i == K``, terminate; else repeat

    Results returned include:

    - Scores from ``scoring`` for each fold
    - Fit and prediction runtimes
    - Optionally: ``y_train``, ``y_test``, ``y_pred``, ``X_train``, ``X_test``
      (if ``return_data=True``)

    A distributed or parallel backend can be chosen using ``backend``.

    Parameters
    ----------
    clusterer : sktime.clustering.BaseClusterer
        Concrete sktime clusterer to benchmark.

    cv : int, sklearn cross-validation generator or an iterable, default=3-fold CV
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None = default = ``KFold(n_splits=3, shuffle=True)``
        - integer, number of folds in a ``KFold`` splitter, ``shuffle=True``
        - An iterable yielding (train, test) splits as arrays of indices.

    X : sktime compatible time series panel data container of Panel scitype
        Panel data container. Supported formats include:

        - ``pd.DataFrame`` with MultiIndex [instance, time] and variable columns
        - 3D ``np.array`` with shape ``[n_instances, n_dimensions, series_length]``
        - Other formats listed in ``datatypes.SCITYPE_REGISTER``

    y : np.ndarray or pd.Series, optional (default=None)
        Ground truth cluster labels. Only required when using external
        metrics (e.g., ``adjusted_rand_score``). For unsupervised evaluation
        with internal metrics (e.g., ``silhouette_score``), ``y`` can be None.

    scoring : callable or list of callables, optional (default=None)
        A scoring function or list of scoring functions. Metrics are
        automatically classified as:

        - **Internal**: ``metric(X, labels)`` — e.g., ``silhouette_score``,
          ``calinski_harabasz_score``, ``davies_bouldin_score``
        - **External**: ``metric(labels_true, labels_pred)`` — e.g.,
          ``adjusted_rand_score``, ``normalized_mutual_info_score``

        If None, defaults to ``silhouette_score``.

    return_data : bool, default=False
        If True, adds columns ``X_train``, ``X_test``, ``y_train``,
        ``y_test``, and ``y_pred`` to the result.

    error_score : {"raise"} or float, default=np.nan
        Value to assign if estimator fitting fails.
        If ``"raise"``, the exception is raised.
        If a numeric value, a ``FitFailedWarning`` is raised
        and the value is assigned.

    backend : {"dask", "loky", "multiprocessing", "threading", "joblib"},
        default=None. Enables parallel evaluation.

        - ``"None"``: sequential loop via list comprehension
        - ``"loky"``, ``"multiprocessing"``, ``"threading"``: ``joblib.Parallel``
        - ``"joblib"``: allows custom joblib backends, e.g., ``spark``
        - ``"dask"``: uses Dask for distributed computation (requires Dask)
        - ``"dask_lazy"``: like ``"dask"``, returns lazy
          ``dask.dataframe.DataFrame``

    backend_params : dict, optional
        Additional parameters passed to the backend.
        Depends on the value of ``backend``:

        - For ``"None"``: ignored
        - For ``"loky"``, ``"multiprocessing"``, ``"threading"``:
          valid ``joblib.Parallel`` params (e.g., ``n_jobs``).
        - For ``"joblib"``: user must include ``backend`` key in this dict.
        - For ``"dask"``: passed to ``dask.compute()``

        Recommendation: Use ``"dask"`` or ``"loky"`` for parallel evaluate.
        ``"threading"`` is unlikely to see speed ups due to the GIL.

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame indexed by fold with the following columns:

        - ``test_<score_name>``: float performance score(s)
        - ``fit_time``: time in seconds to fit the clusterer
        - ``internal_time``: time in seconds to predict (internal metrics)
        - ``external_time``: time in seconds to predict (external metrics)
        - ``X_train``, ``X_test``: panel data (if ``return_data=True``)
        - ``y_train``, ``y_test``: labels (if ``return_data=True``)
        - ``y_pred``: cluster predictions (if ``return_data=True``)

    Examples
    --------
    >>> from sktime.utils._testing.panel import make_clustering_problem
    >>> from sktime.clustering.model_evaluation import evaluate
    >>> from sklearn.model_selection import KFold
    >>> from sktime.clustering.k_means import TimeSeriesKMeans  # doctest: +SKIP
    >>> X = make_clustering_problem(n_instances=20, random_state=42)
    >>> clusterer = TimeSeriesKMeans(n_clusters=3)  # doctest: +SKIP
    >>> cv = KFold(n_splits=3, shuffle=False)
    >>> results = evaluate(  # doctest: +SKIP
    ...     clusterer=clusterer, cv=cv, X=X
    ... )
    """
    from sktime.clustering.base import BaseClusterer

    if not isinstance(clusterer, BaseClusterer):
        raise TypeError(
            f"Expected clusterer to be an instance of BaseClusterer, "
            f"got {type(clusterer)} instead."
        )

    if backend in ["dask", "dask_lazy"]:
        if not _check_soft_dependencies("dask", severity="none"):
            raise RuntimeError(
                "running evaluate with backend='dask' requires the dask package "
                "installed, but dask is not present in the python environment"
            )

    scoring = _check_scores(scoring)

    # Check if external metrics are used but y is None
    if "external" in scoring and y is None:
        ext_names = [m.__name__ for m in scoring["external"]]
        raise ValueError(
            f"External metrics {ext_names} require ground truth labels (y), "
            f"but y=None was provided. Either provide y or use internal "
            f"metrics like silhouette_score."
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

    # Validate and convert X
    X_valid, _, X_metadata = check_is_scitype(X, scitype="Panel", return_metadata=[])
    if not X_valid:
        raise TypeError(f"Expected X dtype Panel. Got {type(X)} instead.")
    X_mtype = X_metadata.get("mtype", None)
    X = convert(X, from_type=X_mtype, to_type=PANDAS_MTYPES)

    # Validate and convert y if provided
    if y is not None:
        y_valid, _, y_metadata = check_is_scitype(
            y, scitype="Table", return_metadata=[]
        )
        if not y_valid:
            raise TypeError(f"Expected y dtype Table. Got {type(y)} instead.")
        y_mtype = y_metadata.get("mtype", None)
        y = convert(y, from_type=y_mtype, to_type="pd_DataFrame_Table")

    _evaluate_fold_kwargs = {
        "clusterer": clusterer,
        "scoring": scoring,
        "return_data": return_data,
        "error_score": error_score,
    }

    def gen_X_y_train_test(X, y, cv):
        """Generate joint splits of X, y as per cv.

        Yields
        ------
        X_train : i-th train split of X as per cv
        X_test : i-th test split of X as per cv
        y_train : i-th train split of y as per cv (None if y is None)
        y_test : i-th test split of y as per cv (None if y is None)
        """
        instance_idx = X.index.get_level_values(0).unique()

        for train_instance_idx, test_instance_idx in cv.split(instance_idx):
            train_instances = instance_idx[train_instance_idx]
            test_instances = instance_idx[test_instance_idx]

            X_train = X.loc[X.index.get_level_values(0).isin(train_instances)]
            X_test = X.loc[X.index.get_level_values(0).isin(test_instances)]

            if y is not None:
                y_train = y.iloc[train_instance_idx]
                y_test = y.iloc[test_instance_idx]
            else:
                y_train = None
                y_test = None

            yield X_train, X_test, y_train, y_test

    # generator for X and y splits
    xy_splits = gen_X_y_train_test(X, y, _cv)

    results = parallelize(
        fun=_evaluate_fold,
        iter=enumerate(xy_splits),
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
