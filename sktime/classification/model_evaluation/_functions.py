#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating classification models."""

__author__ = ["ksharma6", "jgyasu"]
__all__ = ["evaluate"]

import time
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from sktime.datatypes import check_is_scitype, convert
from sktime.exceptions import FitFailedWarning
from sktime.split import InstanceSplitter
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.multiindex import apply_split
from sktime.utils.parallel import parallelize
from sktime.utils.validation.forecasting import check_scoring

PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]


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
    metric_types: dict, return_data: bool = True, cutoff_dtype=None, old_naming=True
) -> dict:
    """Get the ordered column name and input datatype of results."""
    others_metadata = {
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
    metrics_metadata.update(fit_metadata)
    return metrics_metadata.copy()


def _get_pred_args_from_metric(scitype, metric):
    pred_args = {
        "pred_quantiles": "alpha",
    }
    if scitype in pred_args.keys():
        val = getattr(metric, pred_args[scitype], None)
        if val is not None:
            return {pred_args[scitype]: val}
    return {}


def _evaluate_fold(x, meta):
    # unpack args
    i, (y_train, y_test, X_train, X_test) = x
    classifier = meta["classifier"]
    scoring = meta["scoring"]
    return_data = meta["return_data"]
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

    try:
        # fit
        start_fit = time.perf_counter()

        classifier = classifier.clone()
        classifier.fit(y=y_train, X=X_train)

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
            method = getattr(classifier, pred_type[scitype])
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
                    y_pred = method(X_test, **pred_args)
                    pred_time = time.perf_counter() - start_pred
                    temp_result[time_key] = [pred_time]
                    y_preds_cache[y_pred_key] = [y_pred]
                else:
                    y_pred = y_preds_cache[y_pred_key][0]

                score = metric(y_test, y_pred, y_train=y_train)
                temp_result[result_key] = [score]

        # get cutoff
        cutoff = classifier.cutoff

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
                In evaluate, fitting of classifier {type(classifier).__name__} failed,
                you can set error_score='raise' in evaluate to see
                the exception message.
                Fit failed for the {i}-th data split, on training data y_train with
                cutoff {cutoff}, and len(y_train)={len(y_train)}.
                The score will be set to {error_score}.
                Failed classifier with parameters: {classifier}.
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

    temp_result["cutoff"] = [cutoff_ind]
    if return_data:
        temp_result["y_train"] = [y_train]
        temp_result["y_test"] = [y_test]
        temp_result.update(y_preds_cache)
    result = pd.DataFrame(temp_result)
    result = result.astype({"cutoff": cutoff_dtype})
    if old_naming:
        result = result.rename(columns=old_name_mapping)
    column_order = _get_column_order_and_datatype(
        scoring, return_data, cutoff_dtype, old_naming=old_naming
    )
    result = result.reindex(columns=column_order.keys())

    return result, classifier


def evaluate(
    classifier,
    cv=KFold(n_splits=3, shuffle=False),
    X=None,
    y=None,
    scoring: Optional[Union[callable, list[callable]]] = None,
    return_data: bool = False,
    error_score: Union[str, int, float] = np.nan,
    backend: Optional[str] = None,
    backend_params: Optional[dict] = None,
):
    r"""Evaluate classifier using timeseries cross-validation.

    All-in-one statistical performance benchmarking utility for classifiers
    which runs a simple backtest experiment and returns a summary pd.DataFrame.

    The experiment run is the following:

    Train and test folds are generated by ``cv.split(X, y)``.
    For each fold ``i`` in ``K`` number of folds:
    1. Fit the ``classifier`` to the training set ``S_i``
    2. Use the ``classifier`` to make a prediction ``y_pred`` on the  test set ``T_i``
    3. Compute the score using ``scoring(T_i, y_pred)``
    4. If ``i == K``, terminate, otherwise
    5. Set ``i = i + 1`` and go to 1

    Results returned in this function's return are:

    * results of ``scoring`` calculations, from 4,  in the ``i``-th loop
    * runtimes for fitting and/or predicting, from 2, 3, in the ``i``-th loop
    * cutoff state of ``classifier``, at 3, in the ``i``-th loop
    * :math:`y_{train, i}`, :math:`y_{test, i}`, ``y_pred`` (optional)

    A distributed and-or parallel back-end can be chosen via the ``backend`` parameter.

    Parameters
    ----------
    classifier : sktime BaseClassifier descendant (concrete classifier)
         sktime classifier to benchmark
    cv : sklearn KFold
        Provides train/test indices to split data in train/test sets.
       Splits the dataset into k consecutive folds (without shuffling by default).
    X : sktime compatible time series panel data container, Panel scitype, e.g.,
             pd-multiindex: pd.DataFrame with columns = variables,
             index = pd.MultiIndex with first level = instance indices,
             second level = time indices
             numpy3D: 3D np.array (any number of dimensions, equal length series)
             of shape [n_instances, n_dimensions, series_length]
             or of any other supported Panel mtype
             for list of mtypes, see datatypes.SCITYPE_REGISTER
             for specifications, see examples/AA_datatypes_and_datasets.ipynb
    y : sktime compatible tabular data container, Table scitype
         numpy1D iterable, of shape [n_instances]
    scoring : subclass of sklearn.metrics or list of same,
         default=None. Used to get a score function that takes y_pred and y_test
         arguments and accept y_train as keyword argument.
         If None, then uses scoring = accuracy_score().
    return_data : bool, default=False
         Returns three additional columns in the DataFrame, by default False.
         The cells of the columns contain each a pd.Series for y_train,
         y_pred, y_test.
    error_score : "raise" or numeric, default=np.nan
         Value to assign to the score if an exception occurs in estimator fitting.
         If set to "raise", the exception is raised. If a numeric value is given,
         FitFailedWarning is raised.
    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
         Runs parallel evaluate if specified and ``strategy`` is set as "refit".

         - "None": executes loop sequentally, simple list comprehension
         - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
         - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
         - "dask": uses ``dask``, requires ``dask`` package in environment
         - "dask_lazy": same as "dask",
           but changes the return to (lazy) ``dask.dataframe.DataFrame``.

         Recommendation: Use "dask" or "loky" for parallel evaluate.
         "threading" is unlikely to see speed ups due to the GIL and the serialization
         backend (``cloudpickle``) for "dask" and "loky" is generally more robust
         than the standard ``pickle`` library used in "multiprocessing".

    backend_params : dict, optional
         additional parameters passed to the backend as config.
         Directly passed to ``utils.parallel.parallelize``.
         Valid keys depend on the value of ``backend``:

         - "None": no additional parameters, ``backend_params`` is ignored
         - "loky", "multiprocessing" and "threading": default ``joblib`` backends
           any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
           with the exception of ``backend`` which is directly controlled by
           ``backend``.
           If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
           will default to ``joblib`` defaults.
         - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
           any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
           ``backend`` must be passed as a key of ``backend_params`` in this case.
           If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
           will default to ``joblib`` defaults.
         - "dask": any valid keys for ``dask.compute`` can be passed,
           e.g., ``scheduler``

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the classifier.
        Row index is TimeSeriesSplit index of train/test fold in ``cv``.
        Entries in the i-th row are for the i-th train/test split in ``cv``.
        Columns are as follows:

         - test_{scoring.name}: (float) Model performance score. If ``scoring`` is a
         list,
         then there is a column withname ``test_{scoring.name}`` for each scorer.

         - fit_time: (float) Time in sec for ``fit`` or ``update`` on train fold.
         - pred_time: (float) Time in sec to ``predict`` from fitted estimator.
         - y_train: (pd.Series) only present if see ``return_data=True``
         train fold of the i-th split in ``cv``, used to fit/update the classifier.

         - y_pred: (pd.Series) present if see ``return_data=True``
         classifies from fitted classifier for the i-th test fold indices of ``cv``.

         - y_test: (pd.Series) present if see ``return_data=True``
         testing fold of the i-th split in ``cv``, used to compute the metric.

    Examples
    --------
        >>> from sktime.datasets import load_unit_test
        >>> from sktime.classification.model_evaluation import evaluate
        >>> from sklearn.model_selection import KFold
        >>> from sktime.classification.dummy import DummyClassifier
        >>> X, y = load_unit_test()
        >>> classifier = DummyClassifier(strategy="prior")
        >>> cv = KFold(n_splits=3, shuffle=False)
        >>> results = evaluate(classifier=classifier, cv=cv, X=X, y=y)
    """
    if backend in ["dask", "dask_lazy"]:
        if not _check_soft_dependencies("dask", severity="none"):
            raise RuntimeError(
                "running evaluate with backend='dask' requires the dask package "
                "installed, but dask is not present in the python environment"
            )
    scoring = _check_scores(scoring)

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

    cutoff_dtype = str(y.index.dtype)
    _evaluate_fold_kwargs = {
        "classifier": classifier,
        "scoring": scoring,
        "return_data": return_data,
        "error_score": error_score,
        "cutoff_dtype": cutoff_dtype,
    }

    def gen_y_X_train_test(y, X, cv):
        """Generate joint splits of y, X as per cv.

        Yields
        ------
        y_train : i-th train split of y as per cv
        y_test : i-th test split of y as per cv
        X_train : i-th train split of y as per cv.
        X_test : i-th test split of y as per cv.
        """
        splitter = InstanceSplitter(cv)

        genx = splitter.split(X)

        for train_idx, test_idx in genx:
            y_train = apply_split(y, train_idx)
            y_test = apply_split(y, test_idx)
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            yield y_train, y_test, X_train, X_test

    # generator for y and X splits to iterate over below
    yx_splits = gen_y_X_train_test(y, X, cv)

    # Run temporal cross-validation sequentially
    results = []
    for x in enumerate(yx_splits):
        result, classifier = _evaluate_fold(x, _evaluate_fold_kwargs)
        _evaluate_fold_kwargs["classifier"] = classifier
        results.append(result)

    if backend == "dask":
        backend_in = "dask_lazy"
    else:
        backend_in = backend

    results = parallelize(
        fun=_evaluate_fold,
        iter=enumerate(yx_splits),
        meta=_evaluate_fold_kwargs,
        backend=backend_in,
        backend_params=backend_params,
    )

    # final formatting of dask dataframes
    if backend in ["dask", "dask_lazy"]:
        import dask.dataframe as dd

        metadata = _get_column_order_and_datatype(scoring, return_data, cutoff_dtype)

        results = dd.from_delayed(results, meta=metadata)
        if backend == "dask":
            results = results.compute()
    else:
        results = pd.concat(results)

    # final formatting of results DataFrame
    results = results.reset_index(drop=True)

    return results
