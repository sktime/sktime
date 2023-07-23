#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements functions to be used in evaluating forecasting models."""

__author__ = ["aiwalter", "mloning", "fkiraly", "topher-lo"]
__all__ = ["evaluate"]

import time
import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from sktime.datatypes import check_is_scitype, convert_to
from sktime.exceptions import FitFailedWarning
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation._dependencies import _check_soft_dependencies
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


def _evaluate_window(
    y_train,
    y_test,
    X_train,
    X_test,
    i,
    fh,
    forecaster,
    strategy,
    scoring,
    return_data,
    score_name,
    error_score,
    cutoff_dtype,
):
    # set default result values in case estimator fitting fails
    score = error_score
    fit_time = np.nan
    pred_time = np.nan
    cutoff = pd.Period(pd.NaT) if cutoff_dtype.startswith("period") else pd.NA
    y_pred = pd.NA

    if fh is None:
        fh = _select_fh_from_y(y_test)

    try:
        # fit/update
        start_fit = time.perf_counter()
        if i == 0 or strategy == "refit":
            forecaster = forecaster.clone()
            forecaster.fit(y_train, X_train, fh=fh)
        else:  # if strategy in ["update", "no-update_params"]:
            update_params = strategy == "update"
            forecaster.update(y_train, X_train, update_params=update_params)
        fit_time = time.perf_counter() - start_fit

        pred_type = {
            "pred_quantiles": "predict_quantiles",
            "pred_interval": "predict_interval",
            "pred_proba": "predict_proba",
            None: "predict",
        }
        # predict
        start_pred = time.perf_counter()

        if hasattr(scoring, "metric_args"):
            metric_args = scoring.metric_args
        else:
            metric_args = {}

        if hasattr(scoring, "get_tag"):
            scitype = scoring.get_tag("scitype:y_pred", raise_error=False)
        else:
            # If no scitype exists then metric is not proba and no args needed
            scitype = None

        methodname = pred_type[scitype]
        method = getattr(forecaster, methodname)

        y_pred = method(fh, X_test, **metric_args)
        pred_time = time.perf_counter() - start_pred
        # score
        score = scoring(y_test, y_pred, y_train=y_train)
        # get cutoff
        cutoff = forecaster.cutoff

    except Exception as e:
        if error_score == "raise":
            raise e
        else:
            warnings.warn(
                f"""
                In evaluate, fitting of forecaster {type(forecaster).__name__} failed,
                you can set error_score='raise' in evaluate to see
                the exception message. Fit failed for len(y_train)={len(y_train)}.
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

    result = pd.DataFrame(
        {
            score_name: [score],
            "fit_time": [fit_time],
            "pred_time": [pred_time],
            "len_train_window": [len(y_train)],
            "cutoff": [cutoff_ind],
            "y_train": [y_train if return_data else pd.NA],
            "y_test": [y_test if return_data else pd.NA],
            "y_pred": [y_pred if return_data else pd.NA],
        }
    ).astype({"cutoff": cutoff_dtype})

    # Return forecaster if "update"
    if strategy == "update" or (strategy == "no-update_params" and i == 0):
        return result, forecaster
    else:
        return result


def evaluate(
    forecaster,
    cv,
    y,
    X=None,
    strategy: str = "refit",
    scoring: Optional[Union[callable, List[callable]]] = None,
    return_data: bool = False,
    error_score: Union[str, int, float] = np.nan,
    backend: Optional[str] = None,
    compute: bool = True,
    cv_X=None,
    **kwargs,
):
    r"""Evaluate forecaster using timeseries cross-validation.

    All-in-one statistical performance benchmarking utility for forecasters
    which runs a simple backtest experiment and returns a summary pd.DataFrame.

    The experiment run is the following:

    Denote by :math:`y_{train, 1}, y_{test, 1}, \dots, y_{train, K}, y_{test, K}`
    the train/test folds produced by the generator ``cv.split_series(y)``.
    Denote by :math:`X_{train, 1}, X_{test, 1}, \dots, X_{train, K}, X_{test, K}`
    the train/test folds produced by the generator ``cv_X.split_series(X)``
    (if ``X`` is ``None``, consider these to be ``None`` as well).

    0. set ``i = 1``.
    1. ``fit`` the ``forecaster`` to :math:`y_{train, 1}`, :math:`X_{train, 1}`,
      with a ``fh`` to forecast :math:`y_{test, 1}`.
    2. ``y_pred = forecaster.predict``
      (or ``predict_proba`` or ``predict_quantiles``, depending on ``scoring``)
      with exogeneous data :math:`X_{test, i}`
    3. Compute ``scoring`` on ``y_pred``versus :math:`y_{test, 1}`.
    4. if ``i == K``, terminate, otherwise
    5. set ``i = i + 1``
    6. ingest more data :math:`y_{train, i}`, :math:`X_{train, i}`,
      how depends on ``strategy``:
      * if ``strategy == "refit"``, reset and fit ``forecaster`` via ``fit``,
        on :math:`y_{train, i}`, :math:`X_{train, i}` to forecast :math:`y_{test, i}`
      * if ``strategy == "update"``, update ``forecaster`` via ``update``,
        on :math:`y_{train, i}`, :math:`X_{train, i}` to forecast :math:`y_{test, i}`
      * if ``strategy == "no-update_params"``, forward ``forecaster`` via ``update``,
        with argument ``update_params=False``, to the cutoff of :math:`y_{train, i}`
    7. goto 2

    Results returned in this function's return are:
    * results of ``scoring`` calculations, from 3,  in the `i`-th loop
    * runtimes for fitting and/or predicting, from 1, 2, 6, in the `i`-th loop
    * cutoff state of ``forecaster``, at 2, in the `i`-th loop
    * :math:`y_{train, i}`, :math:`y_{test, i}`, ``y_pred`` (optional)

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
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.
    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
        Runs parallel evaluate if specified and `strategy` is set as "refit".
        - "loky", "multiprocessing" and "threading": uses `joblib` Parallel loops
        - "dask": uses `dask`, requires `dask` package in environment
        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (`cloudpickle`) for "dask" and "loky" is generally more robust than the
        standard `pickle` library used in "multiprocessing".
    compute : bool, default=True
        If backend="dask", whether returned DataFrame is computed.
        If set to True, returns `pd.DataFrame`, otherwise `dask.dataframe.DataFrame`.
    cv_X : sktime BaseSplitter descendant, optional
        determines split of ``X`` into test and train folds
        default is ``X`` being split to identical ``loc`` indices as ``y``
        if passed, must have same number of splits as ``cv``
    **kwargs : Keyword arguments
        Only relevant if backend is specified. Additional kwargs are passed into
        `dask.distributed.get_client` or `dask.distributed.Client` if backend is
        set to "dask", otherwise kwargs are passed into `joblib.Parallel`.

    Returns
    -------
    results : pd.DataFrame or dask.dataframe.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.
        Row index is splitter index of train/test fold in `cv`.
        Entries in the i-th row are for the i-th train/test split in `cv`.
        Columns are as follows:
        - test_{scoring.name}: (float) Model performance score. If `scoring` is a list,
            then there is a column withname `test_{scoring.name}` for each scorer.
        - fit_time: (float) Time in sec for `fit` or `update` on train fold.
        - pred_time: (float) Time in sec to `predict` from fitted estimator.
        - len_train_window: (int) Length of train window.
        - cutoff: (int, pd.Timestamp, pd.Period) cutoff = last time index in train fold.
        - y_train: (pd.Series) only present if see `return_data=True`
          train fold of the i-th split in `cv`, used to fit/update the forecaster.
        - y_pred: (pd.Series) present if see `return_data=True`
          forecasts from fitted forecaster for the i-th test fold indices of `cv`.
        - y_test: (pd.Series) present if see `return_data=True`
          testing fold of the i-th split in `cv`, used to compute the metric.

    Examples
    --------
        The type of evaluation that is done by `evaluate` depends on metrics in
        param `scoring`. Default is `MeanAbsolutePercentageError`.
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="mean", sp=12)
    >>> cv = ExpandingWindowSplitter(initial_window=12, step_length=3,
    ... fh=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv)

        Optionally, users may select other metrics that can be supplied
        by `scoring` argument. These can be forecast metrics of any kind,
        i.e., point forecast metrics, interval metrics, quantile forecast metrics.
        https://www.sktime.net/en/stable/api_reference/performance_metrics.html?highlight=metrics
        To evaluate estimators using a specific metric, provide them to the scoring arg.
    >>> from sktime.performance_metrics.forecasting import MeanAbsoluteError
    >>> loss = MeanAbsoluteError()
    >>> results = evaluate(forecaster=forecaster, y=y, cv=cv, scoring=loss)

        Optionally, users can provide a list of metrics to `scoring` argument.
    >>> from sktime.performance_metrics.forecasting import MeanSquaredError
    >>> results = evaluate(
    ...     forecaster=forecaster,
    ...     y=y,
    ...     cv=cv,
    ...     scoring=[MeanSquaredError(square_root=True), MeanAbsoluteError()],
    ... )

        An example of an interval metric is the `PinballLoss`.
        It can be used with all probabilistic forecasters.
    >>> from sktime.forecasting.naive import NaiveVariance
    >>> from sktime.performance_metrics.forecasting.probabilistic import PinballLoss
    >>> loss = PinballLoss()
    >>> forecaster = NaiveForecaster(strategy="drift")
    >>> results = evaluate(forecaster=NaiveVariance(forecaster),
    ... y=y, cv=cv, scoring=loss)
    """
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        raise RuntimeError(
            "running evaluate with backend='dask' requires the dask package installed,"
            "but dask is not present in the python environment"
        )

    _check_strategy(strategy)
    cv = check_cv(cv, enforce_start_with_window=True)
    if isinstance(scoring, List):
        scoring = [check_scoring(s) for s in scoring]
    else:
        scoring = check_scoring(scoring)

    ALLOWED_SCITYPES = ["Series", "Panel", "Hierarchical"]

    y_valid, _, _ = check_is_scitype(y, scitype=ALLOWED_SCITYPES, return_metadata=True)
    if not y_valid:
        raise TypeError(
            f"Expected y dtype {ALLOWED_SCITYPES!r}. Got {type(y)} instead."
        )

    y = convert_to(y, to_type=PANDAS_MTYPES)

    if X is not None:
        X_valid, _, _ = check_is_scitype(
            X, scitype=ALLOWED_SCITYPES, return_metadata=True
        )
        if not X_valid:
            raise TypeError(
                f"Expected X dtype {ALLOWED_SCITYPES!r}. Got {type(X)} instead."
            )
        X = convert_to(X, to_type=PANDAS_MTYPES)

    score_name = (
        f"test_{scoring.name}"
        if not isinstance(scoring, List)
        else f"test_{scoring[0].name}"
    )
    cutoff_dtype = str(y.index.dtype)
    _evaluate_window_kwargs = {
        "fh": cv.fh,
        "forecaster": forecaster,
        "scoring": scoring if not isinstance(scoring, List) else scoring[0],
        "strategy": strategy,
        "return_data": True,
        "error_score": error_score,
        "score_name": score_name,
        "cutoff_dtype": cutoff_dtype,
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
                from sktime.forecasting.model_selection import (
                    SameLocSplitter,
                    TestPlusTrainSplitter,
                )

                cv_X = SameLocSplitter(TestPlusTrainSplitter(cv), y)

            genx = cv_X.split_series(X)

            for (y_train, y_test), (X_train, X_test) in zip(geny, genx):
                yield y_train, y_test, X_train, X_test

    # generator for y and X splits to iterate over below
    yx_splits = gen_y_X_train_test(y, X, cv, cv_X)

    # dispatch by backend and strategy
    if backend is None or strategy in ["update", "no-update_params"]:
        # Run temporal cross-validation sequentially
        results = []
        for i, (y_train, y_test, X_train, X_test) in enumerate(yx_splits):
            if strategy == "update" or (strategy == "no-update_params" and i == 0):
                result, forecaster = _evaluate_window(
                    y_train,
                    y_test,
                    X_train,
                    X_test,
                    i,
                    **_evaluate_window_kwargs,
                )
                _evaluate_window_kwargs["forecaster"] = forecaster
            else:
                result = _evaluate_window(
                    y_train,
                    y_test,
                    X_train,
                    X_test,
                    i,
                    **_evaluate_window_kwargs,
                )
            results.append(result)
        results = pd.concat(results)

    elif backend == "dask":
        # Use Dask delayed instead of joblib,
        # which uses Futures under the hood
        import dask.dataframe as dd
        from dask import delayed as dask_delayed

        results = []
        for i, (y_train, y_test, X_train, X_test) in enumerate(yx_splits):
            results.append(
                dask_delayed(_evaluate_window)(
                    y_train,
                    y_test,
                    X_train,
                    X_test,
                    i,
                    **_evaluate_window_kwargs,
                )
            )
        results = dd.from_delayed(
            results,
            meta={
                score_name: "float",
                "fit_time": "float",
                "pred_time": "float",
                "len_train_window": "int",
                "cutoff": cutoff_dtype,
                "y_train": "object",
                "y_test": "object",
                "y_pred": "object",
            },
        )
        if compute:
            results = results.compute()

    else:
        # Otherwise use joblib
        from joblib import Parallel, delayed

        results = Parallel(backend=backend, **kwargs)(
            delayed(_evaluate_window)(
                y_train,
                y_test,
                X_train,
                X_test,
                i,
                **_evaluate_window_kwargs,
            )
            for i, (y_train, y_test, X_train, X_test) in enumerate(yx_splits)
        )
        results = pd.concat(results)

    # final formatting of results DataFrame
    results = results.reset_index(drop=True)
    if isinstance(scoring, List):
        for s in scoring[1:]:
            results[f"test_{s.name}"] = np.nan
            for row in results.index:
                results.loc[row, f"test_{s.name}"] = s(
                    results["y_test"].loc[row],
                    results["y_pred"].loc[row],
                    y_train=results["y_train"].loc[row],
                )

    # drop pointer to data if not requested
    if not return_data:
        results = results.drop(columns=["y_train", "y_test", "y_pred"])
    results = results.astype({"len_train_window": int})

    return results
