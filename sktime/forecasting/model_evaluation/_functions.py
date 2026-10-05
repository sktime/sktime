#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating forecasting models."""

__author__ = ["aiwalter", "mloning", "fkiraly", "topher-lo", "hazrulakmal", "jgyasu"]
__all__ = ["evaluate"]

import collections.abc
import time
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.datatypes import check_is_scitype, convert
from sktime.exceptions import FitFailedWarning
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.parallel import parallelize
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


def _check_scores(metrics) -> dict:
    """Validate and coerce to BaseMetric and segregate them based on predict type.

    Parameters
    ----------
    metrics : sktime accepted metrics object or a list of them or None

    Return
    ------
    metrics_type : Dict
        The key is metric types and its value is a list of its corresponding metrics.
    """
    if not isinstance(metrics, list):
        metrics = [metrics]

    metrics_type = {}
    for metric in metrics:
        metric = check_scoring(metric)
        # collect predict type
        if hasattr(metric, "get_tag"):
            scitype = metric.get_tag(
                "scitype:y_pred", raise_error=False, tag_value_default="pred"
            )
        else:  # If no scitype exists then metric is a point forecast type
            scitype = "pred"
        if scitype not in metrics_type.keys():
            metrics_type[scitype] = [metric]
        else:
            metrics_type[scitype].append(metric)
    return metrics_type


def _get_column_order_and_datatype(
    metric_types: dict,
    return_data: bool = True,
    cutoff_dtype=None,
    old_naming=True,
    return_model: bool = False,
    global_mode: bool = False,
) -> dict:
    """Get the ordered column name and input datatype of results."""
    others_metadata = {
        "len_train_window": "int",
        "cutoff": cutoff_dtype,
    }
    if global_mode:
        y_metadata = {
            "y_pretrain": "object",
            "y_train": "object",
            "y_test": "object",
        }
    else:
        y_metadata = {
            "y_train": "object",
            "y_test": "object",
        }
    fit_metadata, metrics_metadata = {"fit_time": "float"}, {}
    for scitype in metric_types:
        for metric in metric_types.get(scitype):
            pred_args = _get_pred_args_from_metric(scitype, metric)
            if pred_args == {} or old_naming:
                time_key = f"{scitype}_time"
                result_key = f"test_{metric.name}"
                y_pred_key = f"y_{scitype}"
            else:
                argval = list(pred_args.values())[0]
                time_key = f"{scitype}_{argval}_time"
                result_key = f"test_{metric.name}_{argval}"
                y_pred_key = f"y_{scitype}_{argval}"
            fit_metadata[time_key] = "float"
            metrics_metadata[result_key] = "float"
            if return_data:
                y_metadata[y_pred_key] = "object"
    fit_metadata.update(others_metadata)
    if return_data:
        fit_metadata.update(y_metadata)
    if return_model:
        fit_metadata["fitted_forecaster"] = "object"
    metrics_metadata.update(fit_metadata)
    return metrics_metadata.copy()


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


def _get_pred_args_from_metric(scitype, metric):
    pred_args = {
        "pred_quantiles": "alpha",
        "pred_interval": "coverage",
    }
    if scitype in pred_args.keys():
        val = getattr(metric, pred_args[scitype], None)
        if val is not None:
            return {pred_args[scitype]: val}
    return {}


def _evaluate_window(x, meta):
    """Evaluate forecaster on a single temporal CV fold.

    Called once per temporal train/test split produced by ``evaluate``
    (one row of the output ``DataFrame``). Fits or updates the forecaster,
    makes predictions, computes metrics, and returns a one-row results
    ``DataFrame``.

    Parameters
    ----------
    x : tuple
        Split index and fold data. Always has the form ``(i, split)``, where
        ``i`` is the temporal fold index passed in from ``evaluate``.
        In global mode, ``i`` resets to ``0`` at the start of each instance
        fold; in local mode, ``i`` is simply the index of the temporal split.

        If ``meta["global_mode"]`` is ``False``, ``split`` is
        ``(y_train, y_test, X_train, X_test)``.

        If ``meta["global_mode"]`` is ``True``, ``split`` is
        ``(y_pretrain, y_train, y_test, X_pretrain, X_train, X_test)``.
        Series are named after the forecaster method they are passed to:

        - ``y_pretrain``, ``X_pretrain``:  ``pretrain``
        - ``y_train``, ``X_train``: ``fit`` / ``update``
        - ``y_test``: scoring; ``X_test``: ``predict``

    meta : dict
        Evaluation configuration, assembled by ``evaluate``. Expected keys:

        - ``global_mode`` : bool, whether global evaluation is active
        - ``fh`` : forecasting horizon; if ``None``, inferred from ``y_test``
        - ``forecaster`` : forecaster instance to evaluate
        - ``strategy`` : {"refit", "update", "no-update_params"}
        - ``scoring`` : dict of metrics grouped by prediction scitype
        - ``return_data`` : bool, include fold data columns in result
        - ``return_model`` : bool, include fitted forecaster in result
        - ``error_score`` : value assigned on fit/predict failure, or ``"raise"``
        - ``cutoff_dtype`` : dtype string for the cutoff column

    Returns
    -------
    result : pd.DataFrame
        One-row DataFrame with metric scores, runtimes, cutoff, and optional
        fold data or fitted forecaster columns.
    forecaster : BaseForecaster, optional
        Returned only if ``strategy == "update"``, or if
        ``strategy == "no-update_params"`` and ``i == 0``. The fitted forecaster
        is passed to the next temporal fold when ``evaluate`` runs sequentially.

    Notes
    -----
    Each call handles exactly one temporal fold with a fixed ``i``. The fit
    branch is taken when ``i == 0`` (first temporal fold of an instance fold,
    or first split overall) or when ``strategy == "refit"``: clone the
    forecaster, optionally ``pretrain`` in global mode, then ``fit``.
    Otherwise ``update`` is called on ``y_train``.

    Predictions are cached per prediction scitype so multiple metrics sharing
    the same scitype reuse a single predict call.
    """
    global_mode = meta["global_mode"]
    # unpack args
    if global_mode:
        i, (y_pretrain, y_train, y_test, X_pretrain, X_train, X_test) = x
    else:
        i, (y_train, y_test, X_train, X_test) = x
    fh = meta["fh"]
    forecaster = meta["forecaster"]
    strategy = meta["strategy"]
    scoring = meta["scoring"]
    return_data = meta["return_data"]
    return_model = meta["return_model"]
    error_score = meta["error_score"]
    cutoff_dtype = meta["cutoff_dtype"]

    # set default result values in case estimator fitting fails
    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    cutoff = pd.Period(pd.NaT) if cutoff_dtype.startswith("period") else pd.NA
    y_pred = pd.NA
    temp_result = dict()
    y_preds_cache = dict()
    old_naming = True
    old_name_mapping = {}
    if fh is None:
        fh = _select_fh_from_y(y_test)

    try:
        # fit/update
        start_fit = time.perf_counter()
        if i == 0 or strategy == "refit":
            forecaster = forecaster.clone()
            if global_mode:
                forecaster.pretrain(y=y_pretrain, X=X_pretrain, fh=fh)
            forecaster.fit(y=y_train, X=X_train, fh=fh)
        else:  # strategy in ["update", "no-update_params"]
            update_params = strategy == "update"
            forecaster.update(y=y_train, X=X_train, update_params=update_params)
        fit_time = time.perf_counter() - start_fit

        # predict based on metrics
        pred_type = {
            "pred_quantiles": "predict_quantiles",
            "pred_interval": "predict_interval",
            "pred_proba": "predict_proba",
            "pred": "predict",
        }
        # cache prediction from the first scitype and reuse it to compute other metrics
        for scitype in scoring:
            method = getattr(forecaster, pred_type[scitype])
            if len(set(map(lambda metric: metric.name, scoring.get(scitype)))) != len(
                scoring.get(scitype)
            ):
                old_naming = False
            for metric in scoring.get(scitype):
                pred_args = _get_pred_args_from_metric(scitype, metric)
                if pred_args == {}:
                    time_key = f"{scitype}_time"
                    result_key = f"test_{metric.name}"
                    y_pred_key = f"y_{scitype}"
                else:
                    argval = list(pred_args.values())[0]
                    time_key = f"{scitype}_{argval}_time"
                    result_key = f"test_{metric.name}_{argval}"
                    y_pred_key = f"y_{scitype}_{argval}"
                    old_name_mapping[f"{scitype}_{argval}_time"] = f"{scitype}_time"
                    old_name_mapping[f"test_{metric.name}_{argval}"] = (
                        f"test_{metric.name}"
                    )
                    old_name_mapping[f"y_{scitype}_{argval}"] = f"y_{scitype}"

                # make prediction
                if y_pred_key not in y_preds_cache.keys():
                    start_pred = time.perf_counter()
                    y_pred = method(fh=fh, X=X_test, **pred_args)
                    pred_time = time.perf_counter() - start_pred
                    temp_result[time_key] = [pred_time]
                    y_preds_cache[y_pred_key] = [y_pred]
                else:
                    y_pred = y_preds_cache[y_pred_key][0]

                # evaluate metrics
                score = metric(y_test, y_pred, y_train=y_train)
                temp_result[result_key] = [score]

        # get cutoff
        cutoff = forecaster.cutoff

    except Exception as e:
        if error_score == "raise":
            raise e
        else:  # assign default value when fitting failed
            for scitype in scoring:
                temp_result[f"{scitype}_time"] = [pred_time]
                if return_data:
                    temp_result[f"y_{scitype}"] = [y_pred]
                for metric in scoring.get(scitype):
                    temp_result[f"test_{metric.name}"] = [score]
            warnings.warn(
                f"""
                In evaluate, fitting of forecaster {type(forecaster).__name__} failed,
                you can set error_score='raise' in evaluate to see
                the exception message.
                Fit failed for the {i}-th data split, on training data y_train with
                cutoff {cutoff}, and len(y_train)={len(y_train)}.
                The score will be set to {error_score}.
                Failed forecaster with parameters: {forecaster}.
                """,
                FitFailedWarning,
                stacklevel=2,
            )

    if pd.isnull(cutoff):
        cutoff_ind = cutoff
    else:
        cutoff_ind = cutoff[0]

    # Storing the remaining evaluate detail
    temp_result["fit_time"] = [fit_time]
    temp_result["len_train_window"] = (
        [len(y_train)]
        if not isinstance(y_train.index, pd.MultiIndex)
        else len(np.unique(y_train.index.get_level_values(-1)))
    )

    temp_result["cutoff"] = [cutoff_ind]
    if return_data:
        temp_result["y_train"] = [y_train]
        temp_result["y_test"] = [y_test]
        if global_mode:
            temp_result["y_pretrain"] = [y_pretrain]
        temp_result.update(y_preds_cache)
    if return_model:
        temp_result["fitted_forecaster"] = [deepcopy(forecaster)]
    result = pd.DataFrame(temp_result)
    result = result.astype({"len_train_window": int, "cutoff": cutoff_dtype})
    if old_naming:
        result = result.rename(columns=old_name_mapping)
    column_order = _get_column_order_and_datatype(
        metric_types=scoring,
        return_data=return_data,
        cutoff_dtype=cutoff_dtype,
        old_naming=old_naming,
        return_model=return_model,
        global_mode=global_mode,
    )
    result = result.reindex(columns=column_order.keys())

    # Return forecaster if "update"
    if strategy == "update" or (strategy == "no-update_params" and i == 0):
        return result, forecaster
    else:
        return result


def gen_y_X_train_test_global(y, X, cv, cv_X, cv_global, cv_global_temporal):
    """Generate joint splits of y, X as per cv, cv_X.

    If X is None, pretrain/train/test splits of X are also None.

    If cv_X is None, will default to SameLocSplitter(cv, y_test_global),
    i.e., X splits have same loc index as y temporal splits.

    Yields
    ------
    i : int
        temporal split index within the current global instance fold ``j``.
        Resets to 0 at the start of each instance fold.
    y_pretrain : j-th global pretrain split of y as per cv_global
    y_train : i-th temporal train split of y as per cv, passed to fit
    y_test : i-th temporal test split of y as per cv, used for scoring
    X_pretrain : j-th global pretrain split of X. None if X was None.
    X_train : i-th temporal train split of X. None if X was None.
    X_test : i-th temporal test split of X. None if X was None.
    """
    from sktime.split import InstanceSplitter, SingleWindowSplitter

    if not isinstance(cv_global, InstanceSplitter):
        cv_global = InstanceSplitter(cv_global)

    if cv_global_temporal is not None:
        assert isinstance(cv_global_temporal, SingleWindowSplitter)

    geny = cv_global.split_series(y)
    if X is None:
        for y_pretrain, y_test_global in geny:
            if cv_global_temporal is not None:
                y_pretrain, _ = next(cv_global_temporal.split_series(y_pretrain))
                _, y_test_global = next(cv_global_temporal.split_series(y_test_global))
            for i, (y_train, y_test) in enumerate(cv.split_series(y_test_global)):
                yield i, (y_pretrain, y_train, y_test, None, None, None)
    else:
        from sktime.split import SameLocSplitter

        genx = SameLocSplitter(cv_global, y).split_series(X)

        for (y_pretrain, y_test_global), (X_pretrain, X_test_global) in zip(geny, genx):
            if cv_global_temporal is not None:
                y_pretrain, _ = next(cv_global_temporal.split_series(y_pretrain))
                X_pretrain, _ = next(cv_global_temporal.split_series(X_pretrain))
                _, y_test_global = next(cv_global_temporal.split_series(y_test_global))
                _, X_test_global = next(cv_global_temporal.split_series(X_test_global))
            if cv_X is None:
                _cv_X = SameLocSplitter(cv, y_test_global)
            else:
                _cv_X = cv_X
            for i, ((y_train, y_test), (X_train, X_test)) in enumerate(
                zip(
                    cv.split_series(y_test_global),
                    _cv_X.split_series(X_test_global),
                )
            ):
                yield i, (y_pretrain, y_train, y_test, X_pretrain, X_train, X_test)


def evaluate(
    forecaster,
    cv,
    y,
    X=None,
    strategy: str = "refit",
    scoring: collections.abc.Callable | list[collections.abc.Callable] | None = None,
    return_data: bool = False,
    error_score: str | int | float = np.nan,
    backend: str | None = None,
    cv_X=None,
    backend_params: dict | None = None,
    return_model: bool = False,
    cv_global=None,
    cv_global_temporal=None,
):
    r"""Evaluate forecaster using timeseries cross-validation.

    All-in-one statistical performance benchmarking utility for forecasters
    which runs a simple backtest experiment and returns a summary pd.DataFrame.

    The experiment run is the following:

    In case of non-global evaluation (cv_global=None):

    Denote by :math:`y_{train, 1}, y_{test, 1}, \dots, y_{train, K}, y_{test, K}`
    the train/test folds produced by the generator ``cv.split_series(y)``.
    Denote by :math:`X_{train, 1}, X_{test, 1}, \dots, X_{train, K}, X_{test, K}`
    the train/test folds produced by the generator ``cv_X.split_series(X)``
    (if ``X`` is ``None``, consider these to be ``None`` as well).

    1. Initialize the counter to ``i = 1``
    2. Fit the ``forecaster`` to :math:`y_{train, 1}`, :math:`X_{train, 1}`,
       with ``fh`` set to the absolute indices of :math:`y_{test, 1}`.
    3. Use the ``forecaster`` to make a prediction ``y_pred`` with the exogenous
        data :math:`X_{test, i}`. Predictions are made using either ``predict``,
        ``predict_proba`` or ``predict_quantiles``, depending on ``scoring``.
    4. Compute the ``scoring`` function on ``y_pred`` versus :math:`y_{test, i}`
    5. If ``i == K``, terminate, otherwise
    6. Set ``i = i + 1``
    7. Ingest more data :math:`y_{train, i}`, :math:`X_{train, i}`,
       how depends on ``strategy``:

      - if ``strategy == "refit"``, reset and fit ``forecaster`` via ``fit``,
        on :math:`y_{train, i}`, :math:`X_{train, i}` to forecast :math:`y_{test, i}`
      - if ``strategy == "update"``, update ``forecaster`` via ``update``,
        on :math:`y_{train, i}`, :math:`X_{train, i}` to forecast :math:`y_{test, i}`
      - if ``strategy == "no-update_params"``, forward ``forecaster`` via ``update``,
        with argument ``update_params=False``, to the cutoff of :math:`y_{train, i}`

    8. Go to 3


    In case of global evaluation (cv_global is not None):

    There are two running indices: ``j`` for the instance splitter
    ``cv_global``, and ``i`` for the temporal splitter ``cv``.

    :math:`y_{pretrain, j}, y_{global_test, j}` are produced by
    ``cv_global.split_series(y)`` and are different time series.
    :math:`y_{global_test, j}` is further split into
    :math:`y_{train, i, j}, y_{test, i, j}` by
    ``cv.split_series(y_test)``.
    Exogenous folds :math:`X_{pretrain, j}`, :math:`X_{train, i, j}`,
    :math:`X_{test, i, j}` are produced analogue.

    For each instance fold ``j`` and temporal fold ``i``:

    1. If ``i == 0`` or ``strategy == "refit"``, clone the ``forecaster``,
       pretrain on :math:`y_{pretrain, j}`, :math:`X_{pretrain, j}`,
       then fit on :math:`y_{train, i, j}`, :math:`X_{train, i, j}`,
       with ``fh`` set to the absolute indices of :math:`y_{test, i, j}`.
    2. Otherwise ingest more data :math:`y_{train, i, j}`,
       :math:`X_{train, i, j}` depending on ``strategy``:

      - if ``strategy == "update"``, update via ``update``,
        with ``update_params=True``
      - if ``strategy == "no-update_params"``, update via ``update``,
        with ``update_params=False``

    3. Predict ``y_pred`` with exogenous data :math:`X_{test, i, j}`.
    4. Compute the ``scoring`` function on ``y_pred`` versus
       :math:`y_{test, i, j}`.

    Results returned in this function's return are:

    * results of ``scoring`` calculations, from 4,  in the ``i``-th loop
    * runtimes for fitting and/or predicting, from 2, 3, 7, in the ``i``-th loop
    * cutoff state of ``forecaster``, at 3, in the ``i``-th loop
    * :math:`y_{train, i}`, :math:`y_{test, i}` (and ``y_pretrain`` in global mode),
      ``y_pred`` (optional)
    * fitted forecaster for each fold (optional)

    A distributed and-or parallel back-end can be chosen via the ``backend`` parameter.

    Parameters
    ----------
    forecaster : sktime BaseForecaster descendant (concrete forecaster)
        sktime forecaster to benchmark

    cv : sktime BaseSplitter descendant
        determines split of ``y`` and possibly ``X`` into test and train folds
        y is always split according to ``cv``, see above

        * if ``cv_X`` is not passed, ``X`` splits are subset to ``loc`` equal to ``y``
        * if ``cv_X`` is passed, ``X`` is split according to ``cv_X``

    y : sktime time series container
        Target (endogeneous) time series used in the evaluation experiment
    X : sktime time series container, of same mtype as y
        Exogenous time series used in the evaluation experiment

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        defines the ingestion mode when the forecaster sees new data when window expands

        * "refit" = forecaster is refitted to each training window
        * "update" = forecaster is updated with training window data,
          in sequence provided
        * "no-update_params" = forecaster is updated via ``update``, with
          ``update_params=False``, to the cutoff of each new training window

    scoring : subclass of sktime.performance_metrics.BaseMetric or list of same,
        default=None. Used to get a score function that takes y_pred and y_test
        arguments and accept y_train as keyword argument.
        If None, then uses scoring = MeanAbsolutePercentageError(symmetric=True).
    return_data : bool, default=False
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.
    return_model : bool, default=False
        If True, returns an additional column 'fitted_forecaster' containing the fitted
        forecaster for each fold.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

    backend : string, by default "None".
        Parallelization backend to use for runs.
        Runs parallel evaluate if specified and ``strategy="refit"``.

        - "None": executes loop sequentially, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
          but changes the return to (lazy) ``dask.dataframe.DataFrame``.
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    cv_X : sktime BaseSplitter descendant, optional
        determines split of ``X`` into test and train folds
        default is ``X`` being split to identical ``loc`` indices as ``y``
        if passed, must have same number of splits as ``cv``

    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

        cv_global:  sklearn splitter, or sktime instance splitter, default=None
            If ``cv_global`` is passed, then global benchmarking is applied, as follows:

            1. The ``cv_global`` splitter is used to split data at instance level,
               into a global pretrain set ``y_pretrain``,
               and a global test set ``y_test_global``. This is index ``j``.
            2. ``cv`` then splits the global test set ``y_test_global``
               temporally, to obtain temporal splits ``y_train``, ``y_test``.
               This is index ``i``.
            3. If ``i == 0`` or ``strategy == "refit"``, the estimator is
               cloned, pretrained on ``y_pretrain``, and fitted on ``y_train``.
               Otherwise it is updated on ``y_train`` according to ``strategy``.
            4. The estimator produces predictions``y_pred``, of ``y_test``.

            Overall, with ``y_pretrain``, ``y_train``, ``y_test`` as above,
            the following evaluation will be applied at the start of each
            instance fold (``i == 0``) and on every fold if
            ``strategy == "refit"``:

            .. code-block:: python

                forecaster.pretrain(y=y_pretrain, fh=cv.fh)
                forecaster.fit(y=y_train, fh=cv.fh)
                y_pred = forecaster.predict()
                metric(y_test, y_pred)

        cv_global_temporal:  SingleWindowSplitter, default=None
            ignored if cv_global is None. If passed, it splits the Panel temporally
            before the instance split from cv_global is applied. This avoids
            temporal leakage in the global evaluation across time series.
            Has to be a SingleWindowSplitter.
            cv is applied on the test set of the combined application of
            cv_global and cv_global_temporal.

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.
        Row index is splitter index of train/test fold in ``cv``.
        Entries in the i-th row are for the i-th train/test split in ``cv``.
        Columns are as follows:

        - test_{scoring.name}: (float) Model performance score. If ``scoring`` is a
        list,
        then there is a column withname ``test_{scoring.name}`` for each scorer.

        - fit_time: (float) Time in sec for ``fit`` or ``update`` on train fold.
        - pred_time: (float) Time in sec to ``predict`` from fitted estimator.
        - len_train_window: (int) Length of train window.
        - cutoff: (int, pd.Timestamp, pd.Period) cutoff = last time index in train fold.

        - y_train: (pd.Series) only present if ``return_data=True``,
        train fold of the i-th split in ``cv``, used to fit/update the forecaster.

        - y_pretrain: (pd.Series) present if ``return_data=True`` and
        ``cv_global`` is passed, global pretrain fold used in ``pretrain``.

        - y_pred: (pd.Series) present if ``return_data=True``,
        forecasts from fitted forecaster for the i-th test fold indices of ``cv``.

        - y_test: (pd.Series) present if ``return_data=True``,
        testing fold of the i-th split in ``cv``, used to compute the metric.

        - fitted_forecaster: (BaseForecaster) present if ``return_model=True``,
        fitted forecaster for the i-th split in ``cv``.

    Examples
    --------
    The type of evaluation that is done by ``evaluate`` depends on metrics in
    param ``scoring``. Default is ``MeanAbsolutePercentageError``.

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()[:24]
    >>> forecaster = NaiveForecaster(strategy="mean", sp=3)
    >>> cv = ExpandingWindowSplitter(initial_window=12, step_length=6, fh=[1, 2, 3])
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv)

    To do global evaluation, provide ``cv_global`` and use forecasters supporting
    pretraining.

    >>> from sklearn.model_selection import KFold
    >>> from sktime.datasets import ForecastingData
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.split import InstanceSplitter, SingleWindowSplitter

    >>> data = ForecastingData(   # doctest: +SKIP
    ...     "australian_electricity_demand_dataset"
    ... ).load("y")

    >>> cv = SingleWindowSplitter(fh=range(1, 48))

    >>> results = evaluate(  # doctest: +SKIP
    ...     TinyTimeMixerForecaster(),
    ...     y=data,
    ...     cv=cv,
    ...     cv_global=InstanceSplitter(KFold(5)),
    ...     cv_global_temporal=SingleWindowSplitter(fh=range(48 * 24)),
    ...     strategy="update",
    ... )

    Optionally, users may select other metrics that can be supplied
    by ``scoring`` argument. These can be forecast metrics of any kind as stated `here
    <https://www.sktime.net/en/stable/api_reference/performance_metrics.html?highlight=metrics>`_
    i.e., point forecast metrics, interval metrics, quantile forecast metrics.
    To evaluate estimators using a specific metric, provide them to the scoring arg.

    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteError
    >>> loss = MeanAbsoluteError()
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv, scoring=loss)

    Optionally, users can provide a list of metrics to ``scoring`` argument.

    >>> from sktime.performance_metrics.forecasting import MeanSquaredError
    >>> results = evaluate(
    ...     forecaster=forecaster,
    ...     y=y,
    ...     cv=cv,
    ...     scoring=[MeanSquaredError(square_root=True), MeanAbsoluteError()],
    ... )

    An example of an interval metric is the ``PinballLoss``.
    It can be used with all probabilistic forecasters.

    >>> from sktime.forecasting.naive import NaiveVariance
    >>> from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
    >>> loss = PinballLoss()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> results = evaluate(forecaster=NaiveVariance(forecaster),
    ... y=y, cv=cv, scoring=loss)

    To return fitted models for each fold, set ``return_model=True``:

    >>> results = evaluate(
    ...     forecaster=forecaster,
    ...     y=y,
    ...     cv=cv,
    ...     scoring=loss,
    ...     return_model=True
    ... )
    >>> fitted_forecaster = results.iloc[0]["fitted_forecaster"]
    """
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

    _check_strategy(strategy)
    cv = check_cv(cv, enforce_start_with_window=True)
    scoring = _check_scores(scoring)
    if cv_global is not None:
        ALLOWED_SCITYPES = ["Panel", "Hierarchical"]
    else:
        ALLOWED_SCITYPES = ["Series", "Panel", "Hierarchical"]

    y_valid, _, y_metadata = check_is_scitype(
        y, scitype=ALLOWED_SCITYPES, return_metadata=[]
    )
    if not y_valid:
        raise TypeError(
            f"Expected y dtype {ALLOWED_SCITYPES!r}. Got {type(y)} instead."
        )
    y_mtype = y_metadata["mtype"]

    y = convert(y, from_type=y_mtype, to_type=PANDAS_MTYPES)

    if X is not None:
        X_valid, _, X_metadata = check_is_scitype(
            X, scitype=ALLOWED_SCITYPES, return_metadata=[]
        )
        if not X_valid:
            raise TypeError(
                f"Expected X dtype {ALLOWED_SCITYPES!r}. Got {type(X)} instead."
            )
        X_mtype = X_metadata["mtype"]

        X = convert(X, from_type=X_mtype, to_type=PANDAS_MTYPES)

    cutoff_dtype = str(y.index.dtype)
    _evaluate_window_kwargs = {
        "fh": cv.fh,
        "forecaster": forecaster,
        "scoring": scoring,
        "strategy": strategy,
        "return_data": return_data,
        "return_model": return_model,
        "error_score": error_score,
        "cutoff_dtype": cutoff_dtype,
        "global_mode": cv_global is not None,
    }

    def gen_y_X_train_test(y, X, cv, cv_X):
        """Generate joint splits of y, X as per cv, cv_X.

        If X is None, train/test splits of X are also None.

        If cv_X is None, will default to
        SameLocSplitter(TestPlusTrainSplitter(cv), y)
        i.e., X splits have same loc index as y splits.

        Yields
        ------
        y_train : i-th train split of y as per cv
        y_test : i-th test split of y as per cv
        X_train : i-th train split of y as per cv_X. None if X was None.
        X_test : i-th test split of y as per cv_X. None if X was None.
        """
        geny = cv.split_series(y)
        if X is None:
            for y_train, y_test in geny:
                yield y_train, y_test, None, None
        else:
            if cv_X is None:
                from sktime.split import SameLocSplitter, TestPlusTrainSplitter

                cv_X = SameLocSplitter(TestPlusTrainSplitter(cv), y)

            genx = cv_X.split_series(X)

            for (y_train, y_test), (X_train, X_test) in zip(geny, genx):
                yield y_train, y_test, X_train, X_test

    # generator for y and X splits to iterate over below.
    # Each item is (i, split), where i is the temporal fold index.
    # In global mode i resets to 0 at the start of each instance fold j.
    if cv_global is not None:
        yx_splits = gen_y_X_train_test_global(
            y, X, cv, cv_X, cv_global=cv_global, cv_global_temporal=cv_global_temporal
        )
    else:
        yx_splits = enumerate(gen_y_X_train_test(y, X, cv, cv_X))

    # sequential strategies cannot be parallelized
    not_parallel = strategy in ["update", "no-update_params"]

    # dispatch by backend and strategy
    if not_parallel:
        # Run temporal cross-validation sequentially
        results = []
        for x in yx_splits:
            i = x[0]
            if strategy == "update" or (strategy == "no-update_params" and i == 0):
                result, forecaster = _evaluate_window(x, _evaluate_window_kwargs)
                _evaluate_window_kwargs["forecaster"] = forecaster
            else:
                result = _evaluate_window(x, _evaluate_window_kwargs)
            results.append(result)
    else:
        if backend == "dask":
            backend_in = "dask_lazy"
        else:
            backend_in = backend
        results = parallelize(
            fun=_evaluate_window,
            iter=yx_splits,
            meta=_evaluate_window_kwargs,
            backend=backend_in,
            backend_params=backend_params,
        )

    # final formatting of dask dataframes
    if backend in ["dask", "dask_lazy"] and not not_parallel:
        import dask.dataframe as dd

        metadata = _get_column_order_and_datatype(
            scoring,
            return_data,
            cutoff_dtype,
            return_model=return_model,
            global_mode=cv_global is not None,
        )

        results = dd.from_delayed(results, meta=metadata)
        if backend == "dask":
            results = results.compute()
    else:
        results = pd.concat(results)

    # final formatting of results DataFrame
    results = results.reset_index(drop=True)

    return results
