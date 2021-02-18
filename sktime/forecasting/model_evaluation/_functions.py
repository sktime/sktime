# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
from tqdm.auto import tqdm
from sktime.utils.validation.forecasting import check_y
from sktime.utils.validation.forecasting import check_cv
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.validation.forecasting import check_scoring

__author__ = ["Martin Walter"]
__all__ = ["evaluate"]


def evaluate(
    forecaster, cv, y, X=None, strategy="refit", scoring=None, return_data=False
):
    """Evaluate forecaster using cross-validation

    Parameters
    ----------
    forecaster : sktime.forecaster
        Any forecaster
    cv : sktime.SlidingWindowSplitter or sktime.ExpandingWindowSplitter
        Splitter of how to split the data into test data and train data
    y : pd.Series
        Target time series to which to fit the forecaster.
    X : pd.DataFrame, optional (default=None)
        Exogenous variables
    strategy : str, optional
        Must be "refit" or "update", by default "refit". The strategy defines
        whether forecaster is only fitted on the first train window data and
        then updated or always refitted.
    scoring : object of class MetricFunctionWrapper from
        sktime.performance_metrics, optional. Example scoring=sMAPE().
        Used to get a score function that takes y_pred and y_test as arguments,
        by default None (if None, uses sMAPE)
    return_data : bool, optional
        Returns three additional columns in the DataFrame, by default False.
        The cells of the columns contain each a pd.Series for y_train,
        y_pred, y_test.

    Returns
    -------
    pd.DataFrame
        DataFrame that contains several columns with information regarding each
        refit/update and prediction of the forecaster.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.performance_metrics.forecasting import evaluate
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_airline()
    >>> forecaster = NaiveForecaster(strategy="drift", sp=12)
    >>> cv = ExpandingWindowSplitter(
        initial_window=24,
        step_length=12,
        fh=[1,2,3,4,5,6,7,8,9,10,11,12]
        )
    >>> evaluate(forecaster=forecaster, y=y, cv=cv)
    """
    cv = check_cv(cv)
    y = check_y(y)
    _check_strategies(strategy)
    scoring = check_scoring(scoring)

    n_splits = cv.get_n_splits(y)
    results = pd.DataFrame()
    cv.start_with_window = True

    for i, (train, test) in enumerate(tqdm(cv.split(y), total=n_splits)):
        # get initial window, if required
        if i == 0 and cv.initial_window and strategy == "update":
            train, test = cv.split_initial(y)
            # this might have to be directly handled in split_initial()
            test = test[: len(cv.fh)]

        # create train/test data
        y_train = y.iloc[train]
        y_test = y.iloc[test]

        X_train = X.iloc[train] if X else None
        X_test = X.iloc[test] if X else None

        # fit/update
        start_fit = time.time()
        if strategy == "refit" or i == 0:
            forecaster.fit(
                y=y_train,
                X=X_train,
                fh=ForecastingHorizon(y_test.index, is_relative=False),
            )
        else:
            # strategy == "update" and i != 0:
            forecaster.update(y=y_train, X=X_train)
        fit_time = time.time() - start_fit

        # predict
        start_pred = time.time()
        y_pred = forecaster.predict(
            fh=ForecastingHorizon(y_test.index, is_relative=False), X=X_test
        )
        pred_time = time.time() - start_pred

        # save results
        results = results.append(
            {
                "test_" + scoring.__class__.__name__: scoring(y_pred, y_test),
                "fit_time": fit_time,
                "pred_time": pred_time,
                "len_train_window": len(y_train),
                "cutoff": forecaster.cutoff,
                "y_train": y_train if return_data else np.nan,
                "y_test": y_test if return_data else np.nan,
                "y_pred": y_pred if return_data else np.nan,
            },
            ignore_index=True,
        )

    # post-processing of results
    if not return_data:
        results = results.drop(columns=["y_train", "y_test", "y_pred"])
    results["len_train_window"] = results["len_train_window"].astype(int)

    return results


def _check_strategies(strategy):
    """Assert strategy value

    Parameters
    ----------
    strategy : str
        strategy of how to evaluate a forecaster

    Raises
    ------
    ValueError
        If strategy value is not in expected values, raise error.
    """
    if strategy not in ["refit", "update"]:
        raise ValueError('strategy must be either "refit" or "update"')
