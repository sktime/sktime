#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating forecasting models."""

__author__ = ["aiwalter", "mloning", "fkiraly", "topher-lo", "hazrulakmal"]
__all__ = ["evaluate"]

import time
import warnings
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import pandas as pd

from sktime.datatypes import check_is_scitype, convert
from sktime.exceptions import FitFailedWarning
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.dependencies import _check_soft_dependencies
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
) -> dict:
    """Get the ordered column name and input datatype of results."""
    others_metadata = {
        "len_train_window": "int",
        "cutoff": cutoff_dtype,
    }
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


# should we remove _split since this is no longer being used?
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
    global_mode = meta["global_mode"]
    # unpack args
    if global_mode:
        i, (y_train, y_hist, y_true, X_train, X_test) = x
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
        fh = _select_fh_from_y(y_true if global_mode else y_test)

    try:
        # fit/update
        start_fit = time.perf_counter()
        if i == 0 or strategy == "refit":
            forecaster = forecaster.clone()
            forecaster.fit(y=y_train, X=X_train, fh=fh)
        else:  # if strategy in ["update", "no-update_params"]:
            update_params = strategy == "update"
            forecaster.update(y_train, X_train, update_params=update_params)
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
                    if global_mode:
                        y_pred = method(fh=fh, X=X_test, y=y_hist, **pred_args)
                    else:
                        y_pred = method(fh=fh, X=X_test, **pred_args)
                    pred_time = time.perf_counter() - start_pred
                    temp_result[time_key] = [pred_time]
                    y_preds_cache[y_pred_key] = [y_pred]
                else:
                    y_pred = y_preds_cache[y_pred_key][0]

                # evaluate metrics
                if global_mode:
                    score = metric(y_true, y_pred, y_train=y_train)
                else:
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
        if global_mode:
            temp_result["y_hist"] = [y_hist]
            temp_result["y_true"] = [y_true]
        else:
            temp_result["y_test"] = [y_test]
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
    )
    result = result.reindex(columns=column_order.keys())

    # Return forecaster if "update"
    if strategy == "update" or (strategy == "no-update_params" and i == 0):
        return result, forecaster
    else:
        return result


def gen_y_X_train_test_global(y, X, cv, cv_X, cv_global, cv_global_temporal):
    """Generate joint splits of y, X as per cv, cv_X.

    If X is None, train/test splits of X are also None.

    If cv_X is None, will default to
    SameLocSplitter(TestPlusTrainSplitter(cv), y)
    i.e., X splits have same loc index as y splits.

    Yields
    ------
    y_train : i-th train split of y as per cv
    y_hist : i-th test history value split of y as per cv
    y_true : i-th test true value split of y as per cv
    X_train : i-th train split of y as per cv_X. None if X was None.
    X_test : i-th test split of y as per cv_X. None if X was None.
    """
    from sktime.split import InstanceSplitter, SingleWindowSplitter

    if not isinstance(cv_global, InstanceSplitter):
        cv_global = InstanceSplitter(cv_global)

    if cv_global_temporal is not None:
        assert isinstance(cv_global_temporal, SingleWindowSplitter)

    geny = cv_global.split_series(y)
    if X is None:
        for y_train, y_test in geny:
            if cv_global_temporal is not None:
                y_train, _ = next(cv_global_temporal.split_series(y_train))
                _, y_test = next(cv_global_temporal.split_series(y_test))
            for y_hist, y_true in cv.split_series(y_test):
                yield y_train, y_hist, y_true, None, None
    else:
        from sktime.split import SameLocSplitter, TestPlusTrainSplitter

        genx = SameLocSplitter(cv_global, y).split_series(X)

        for (y_train, y_test), (X_train, X_test) in zip(geny, genx):
            if cv_global_temporal is not None:
                y_train, _ = next(cv_global_temporal.split_series(y_train))
                X_train, _ = next(cv_global_temporal.split_series(X_train))
                _, y_test = next(cv_global_temporal.split_series(y_test))
                _, X_test = next(cv_global_temporal.split_series(X_test))
            if cv_X is None:
                _cv_X = TestPlusTrainSplitter(SameLocSplitter(cv, y_test))
            else:
                _cv_X = cv_X
            for (y_hist, y_true), (_, X_future) in zip(
                cv.split_series(y_test), _cv_X.split_series(X_test)
            ):
                # X_hist is not used in the evaluation
                yield y_train, y_hist, y_true, X_train, X_future


def evaluate(
    forecaster,
    cv,
    y,
    X=None,
    strategy: str = "refit",
    scoring: Optional[Union[callable, list[callable]]] = None,
    return_data: bool = False,
    error_score: Union[str, int, float] = np.nan,
    backend: Optional[str] = None,
    cv_X=None,
    backend_params: Optional[dict] = None,
    return_model: bool = False,
    cv_global=None,
    cv_global_temporal=None,
):
    r"""Evaluate forecaster using timeseries cross-validation.

    All-in-one statistical performance benchmarking utility for forecasters
    which runs a simple backtest experiment and returns a summary pd.DataFrame.

    The experiment run is the following:

    In  case of non-global evaluation (cv_global=None):

    Denote by :math:`y_{train, 1}, y_{test, 1}, \dots, y_{train, K}, y_{test, K}`
    the train/test folds produced by the generator ``cv.split_series(y)``.
    Denote by :math:`X_{train, 1}, X_{test, 1}, \dots, X_{train, K}, X_{test, K}`
    the train/test folds produced by the generator ``cv_X.split_series(X)``
    (if ``X`` is ``None``, consider these to be ``None`` as well).

    1. Initialize the counter to ``i = 1``
    2. Fit the ``forecaster`` to :math:`y_{train, 1}`, :math:`X_{train, 1}`,
       with ``fh`` set to the absolute indices of :math:`y_{test, 1}`.
    3. Use the ``forecaster`` to make a prediction ``y_pred`` with the exogeneous
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

    The test folds are denoted by :math:`y_{train, 1}, y_{hist, 1}, y_{true, 1},
    \dots, y_{train, K}`, y_{hist, K}, y_{true, K}`.
    :math:`y_{train, i}, y_{test, i}`` are produced by the generator
    ``cv_global.split_series(y)``. Whereby :math:`y_{train, i}, y_{test, i}`
    are different time series.
    :math:`y_{test, i}` is further split into :math:`y_{hist, i}, y_{true, i}` by
    the generator ``cv.split_series(y_test)``.
    Denote by :math:`X_{train, 1}, X_{hist, 1}, X_{true, 1}, \dots, X_{train, K},
    X_{hist, K}, X_{true, K}` the train/test folds are produced analogue by
    ``cv_global.split_series(X)`` and ``cv.split_series(X_test)``.

    1. Initialize the counter to ``i = 1``
    2. Fit the ``forecaster`` to :math:`y_{train, i}`, :math:`X_{train, 1i`,
       with ``fh`` set to the absolute indices of :math:`y_{true, i}`.
    3. Use the ``forecaster`` to make a prediction ``y_pred`` with the exogeneous
        data :math:`X_{true, i}` and the historical values :math:`y{hist, i}`.
        Predictions are made using either ``predict``,
        ``predict_proba`` or ``predict_quantiles``, depending on ``scoring``.
    4. Compute the ``scoring`` function on ``y_pred`` versus :math:`y_{true, i}`
    5. If ``i == K``, terminate, otherwise
    6. Set ``i = i + 1``
    8. Go to 2

    Results returned in this function's return are:

    * results of ``scoring`` calculations, from 4,  in the ``i``-th loop
    * runtimes for fitting and/or predicting, from 2, 3, 7, in the ``i``-th loop
    * cutoff state of ``forecaster``, at 3, in the ``i``-th loop
    * :math:`y_{train, i}`, :math:`y_{test, i}`, ``y_pred`` (optional)
    * fitted forecaster for each fold (optional)

    A distributed and-or parallel back-end can be chosen via the ``backend`` parameter.

    Parameters
    ----------
    forecaster : sktime BaseForecaster descendant (concrete forecaster)
        sktime forecaster to benchmark
    cv : sktime BaseSplitter descendant
        determines split of ``y`` and possibly ``X`` into test and train folds
        y is always split according to ``cv``, see above
        if ``cv_X`` is not passed, ``X`` splits are subset to ``loc`` equal to ``y``
        if ``cv_X`` is passed, ``X`` is split according to ``cv_X``
    y : sktime time series container
        Target (endogeneous) time series used in the evaluation experiment
    X : sktime time series container, of same mtype as y
        Exogenous time series used in the evaluation experiment
    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
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

            1. the ``cv_global`` splitter is used to split data at instance level,
            into a global training set ``y_train``,
            and a global test set ``y_test_global``.
            2. The estimator is fitted to the global training set ``y_train``.
            3. ``cv_splitter`` then splits the global test set ``y_test_global``
            temporally, to obtain temporal splits ``y_past``, ``y_true``.

            Overall, with ``y_train``, ``y_past``, ``y_true`` as above,
            the following evaluation will be applied:

            .. code-block:: python

                forecaster.fit(y=y_train, fh=cv.fh)
                y_pred = forecaster.predict(y=y_past)
                metric(y_true, y_pred)
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

    # generator for y and X splits to iterate over below
    if cv_global is not None:
        yx_splits = gen_y_X_train_test_global(
            y, X, cv, cv_X, cv_global=cv_global, cv_global_temporal=cv_global_temporal
        )
    else:
        yx_splits = gen_y_X_train_test(y, X, cv, cv_X)

    # sequential strategies cannot be parallelized
    not_parallel = strategy in ["update", "no-update_params"]

    # dispatch by backend and strategy
    if not_parallel:
        # Run temporal cross-validation sequentially
        results = []
        for x in enumerate(yx_splits):
            is_first = x[0] == 0  # first iteration
            if strategy == "update" or (strategy == "no-update_params" and is_first):
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
            iter=enumerate(yx_splits),
            meta=_evaluate_window_kwargs,
            backend=backend_in,
            backend_params=backend_params,
        )

    # final formatting of dask dataframes
    if backend in ["dask", "dask_lazy"] and not not_parallel:
        import dask.dataframe as dd

        metadata = _get_column_order_and_datatype(
            scoring, return_data, cutoff_dtype, return_model=return_model
        )

        results = dd.from_delayed(results, meta=metadata)
        if backend == "dask":
            results = results.compute()
    else:
        results = pd.concat(results)

    # final formatting of results DataFrame
    results = results.reset_index(drop=True)

    return results
