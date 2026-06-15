#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Utilities for performance evaluation of time series regression models."""

__author__ = ["jgyasu"]
__all__ = ["evaluate"]

import collections.abc
import time
import warnings

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datatypes import check_is_scitype, convert
from sktime.exceptions import FitFailedWarning
from sktime.utils.parallel import parallelize

PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


def _check_scores(metrics):
    """Validate regression metrics and return them in a standardised dict.

    For regression, all metrics are deterministic (no predict_proba), so
    every metric is placed under the ``"pred"`` key.

    Parameters
    ----------
    metrics : callable, list of callables, or None
        Regression scoring functions that accept ``(y_true, y_pred)``.
        If None, defaults to ``sklearn.metrics.mean_squared_error``.

    Returns
    -------
    metrics_type : dict
        Dictionary with key ``"pred"`` mapping to the list of metrics.
    """
    if metrics is None:
        from sklearn.metrics import mean_squared_error

        metrics = [mean_squared_error]

    if not isinstance(metrics, list):
        metrics = [metrics]

    for metric in metrics:
        if not callable(metric):
            raise ValueError(f"Metric {metric} is not callable")

    return {"pred": metrics}


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
            time_key = f"{scitype}_time"
            result_key = f"test_{metric.__name__}"
            y_pred_key = f"y_{scitype}"
            fit_metadata[time_key] = "float"
            metrics_metadata[result_key] = "float"
            if return_data:
                y_metadata[y_pred_key] = "object"
    if return_data:
        fit_metadata.update(y_metadata)
    metrics_metadata.update(fit_metadata)
    return metrics_metadata.copy()


def _evaluate_fold(x, meta):
    """Evaluate a single CV fold for a regressor.

    Parameters
    ----------
    x : tuple of (int, tuple)
        ``(fold_index, (y_train, y_test, X_train, X_test))``.
    meta : dict
        Must contain keys ``"regressor"``, ``"scoring"``, ``"return_data"``,
        and ``"error_score"``.

    Returns
    -------
    result : pd.DataFrame
        Single-row DataFrame with scores, timings, and optionally data.
    """
    i, (y_train, y_test, X_train, X_test) = x
    regressor = meta["regressor"]
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

        regressor = regressor.clone()
        regressor.fit(y=y_train, X=X_train)

        fit_time = time.perf_counter() - start_fit

        # predict — regressors only have deterministic predict
        for scitype in scoring:
            for metric in scoring.get(scitype):
                time_key = f"{scitype}_time"
                result_key = f"test_{metric.__name__}"
                y_pred_key = f"y_{scitype}"

                # make prediction (cache to avoid duplicate predict calls)
                if y_pred_key not in y_preds_cache.keys():
                    start_pred = time.perf_counter()
                    y_pred = regressor.predict(X_test)
                    pred_time = time.perf_counter() - start_pred
                    temp_result[time_key] = [pred_time]
                    y_preds_cache[y_pred_key] = [y_pred]
                else:
                    y_pred = y_preds_cache[y_pred_key][0]

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


def evaluate(
    regressor,
    cv=None,
    X=None,
    y=None,
    scoring: collections.abc.Callable | list[collections.abc.Callable] | None = None,
    return_data: bool = False,
    error_score: str | int | float = np.nan,
    backend: str | None = None,
    backend_params: dict | None = None,
):
    r"""Evaluate regressor using cross-validation.

    All-in-one statistical performance benchmarking utility for time series
    regressors, which runs a simple backtest experiment and returns a summary
    DataFrame.

    The experiment runs the following:

    1. Train and test folds are generated by ``cv.split(X, y)``.
    2. For each fold ``i`` in ``K`` folds:

        a. Fit the ``regressor`` on the training set ``S_i``
        b. Predict ``y_pred`` on the test set ``T_i``
        c. Compute the score via ``scoring(T_i, y_pred)``
        d. If ``i == K``, terminate; else repeat

    Results returned include:

    - Scores from ``scoring`` for each fold
    - Fit and prediction runtimes
    - Optionally: ``y_train``, ``y_test``, and ``y_pred``
      (if ``return_data=True``)

    A distributed or parallel backend can be chosen using ``backend``.

    Parameters
    ----------
    regressor : sktime BaseRegressor
        Concrete sktime regressor to benchmark.

    cv : int, sklearn cross-validation generator or an iterable, default=3-fold CV
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None = default = ``KFold(n_splits=3, shuffle=True)``
        - integer, number of folds in a ``KFold`` splitter, ``shuffle=True``
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, :class:`KFold` is used with
        ``shuffle=True`` so the splits will be randomized.

    X : sktime-compatible panel data (Panel scitype)
        Panel data container. Supported formats include:

        - ``pd.DataFrame`` with MultiIndex [instance, time] and variable columns
        - 3D ``np.array`` with shape ``[n_instances, n_dimensions, series_length]``
        - Other formats listed in ``datatypes.SCITYPE_REGISTER``

    y : sktime-compatible tabular data (Table scitype)
        Target variable, typically a 1D ``np.ndarray`` or ``pd.Series``
        of shape ``[n_instances]`` with continuous (float) values.

    scoring : callable or list of callables, optional (default=None)
        A scoring function or list of scoring functions that take
        ``(y_true, y_pred)``.
        If None, defaults to ``mean_squared_error``.

    return_data : bool, default=False
        If True, adds columns ``y_train``, ``y_test``, and
        ``y_pred`` (as pd.Series) to the result.

    error_score : {"raise"} or float, default=np.nan
        Value to assign if estimator fitting fails.
        If "raise", the exception is raised.
        If a numeric value, a ``FitFailedWarning`` is raised
        and the value is assigned.

    backend : {"dask", "loky", "multiprocessing", "threading", "joblib"}, default=None
        Enables parallel evaluation.

        - "None": sequential loop via list comprehension
        - "loky", "multiprocessing", "threading": uses ``joblib.Parallel``
        - "joblib": allows custom joblib backends, e.g., ``spark``
        - "dask": uses Dask for distributed computation (requires Dask)
        - "dask_lazy": like "dask", but returns a lazy ``dask.dataframe.DataFrame``

    backend_params : dict, optional
        Additional parameters passed to the backend.
        Depends on the value of ``backend``:

        - For "None": ignored
        - For "loky", "multiprocessing", "threading":
          valid ``joblib.Parallel`` params (e.g., ``n_jobs``).
          ``backend`` is controlled by this function and should not be included.
        - For "joblib": user must include ``backend`` key in this dictionary.
          Also accepts ``n_jobs`` and other ``joblib.Parallel`` args.
        - For "dask": passed to ``dask.compute()``, e.g., ``scheduler="threads"``

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the
        serialization backend (``cloudpickle``) for "dask" and "loky" is
        generally more robust than the standard ``pickle`` library used
        in "multiprocessing".

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame indexed by fold (i-th split in ``cv``) with the following columns:

        - ``test_<score_name>``: float performance score(s)
        - ``fit_time``: time in seconds to fit the regressor
        - ``pred_time``: time in seconds to predict from the regressor
        - ``y_train``: pd.Series of train targets (if ``return_data=True``)
        - ``y_pred``: pd.Series of predictions (if ``return_data=True``)
        - ``y_test``: pd.Series of test targets (if ``return_data=True``)

    Examples
    --------
    >>> from sktime.datasets import load_unit_test
    >>> from sktime.regression.model_evaluation import evaluate
    >>> from sklearn.model_selection import KFold
    >>> from sktime.regression.dummy import DummyRegressor
    >>> from sktime.utils._testing.panel import make_regression_problem
    >>> X, y = make_regression_problem()
    >>> regressor = DummyRegressor(strategy="mean")
    >>> cv = KFold(n_splits=3, shuffle=False)
    >>> results = evaluate(regressor=regressor, cv=cv, X=X, y=y)
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

    _evaluate_fold_kwargs = {
        "regressor": regressor,
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
