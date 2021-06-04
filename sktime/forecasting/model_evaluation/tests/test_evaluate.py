#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Martin Walter", "Markus Löning"]
__all__ = [
    "test_evaluate_common_configs",
    "test_evaluate_initial_window",
    "test_evaluate_no_exog_against_with_exog",
]

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sktime.datasets import load_longley
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.tests._config import TEST_FHS
from sktime.forecasting.tests._config import TEST_STEP_LENGTHS
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
)
from sktime.utils._testing.forecasting import make_forecasting_problem


def _check_evaluate_output(out, cv, y, scoring):
    assert isinstance(out, pd.DataFrame)

    # Check column names.
    assert set(out.columns) == {
        "cutoff",
        "fit_time",
        "len_train_window",
        "pred_time",
        f"test_{scoring.name}",
    }

    # Check number of rows against number of splits.
    n_splits = cv.get_n_splits(y)
    assert out.shape[0] == n_splits

    # Check if all timings are positive.
    assert np.all(out.filter(like="_time") >= 0)

    # Check cutoffs.
    np.testing.assert_array_equal(
        out["cutoff"].to_numpy(), y.iloc[cv.get_cutoffs(y)].index.to_numpy()
    )

    # Check training window lengths.
    if isinstance(cv, SlidingWindowSplitter) and cv.initial_window is not None:
        assert np.all(out.loc[0, "len_train_window"] == cv.initial_window)
        assert np.all(out.loc[1:, "len_train_window"] == cv.window_length)

    elif isinstance(cv, ExpandingWindowSplitter):
        step = cv.step_length
        start = cv.window_length
        end = start + (n_splits * step)
        expected = np.arange(start, end, step)
        actual = out.loc[:, "len_train_window"].to_numpy()

        np.testing.assert_array_equal(expected, actual)
        assert np.all(out.loc[0, "len_train_window"] == cv.window_length)

    else:
        assert np.all(out.loc[:, "len_train_window"] == cv.window_length)


# Test using MAPE and MASE scorers so that tests cover a metric that doesn't
# use y_train (MAPE) and one that does use y_train (MASE).
@pytest.mark.parametrize("CV", [SlidingWindowSplitter, ExpandingWindowSplitter])
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", [7, 10])
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS)
@pytest.mark.parametrize("strategy", ["refit", "update"])
@pytest.mark.parametrize(
    "scoring",
    [
        MeanAbsolutePercentageError(symmetric=True),
        MeanAbsoluteScaledError(),
    ],
)
def test_evaluate_common_configs(CV, fh, window_length, step_length, strategy, scoring):
    y = make_forecasting_problem(n_timepoints=30, index_type="int")
    forecaster = NaiveForecaster()
    cv = CV(fh, window_length, step_length=step_length)

    out = evaluate(
        forecaster=forecaster, y=y, cv=cv, strategy=strategy, scoring=scoring
    )
    _check_evaluate_output(out, cv, y, scoring)

    # check scoring
    actual = out.loc[:, f"test_{scoring.name}"]

    n_splits = cv.get_n_splits(y)
    expected = np.empty(n_splits)
    for i, (train, test) in enumerate(cv.split(y)):
        f = clone(forecaster)
        f.fit(y.iloc[train], fh=fh)
        expected[i] = scoring(y.iloc[test], f.predict(), y_train=y.iloc[train])

    np.testing.assert_array_equal(actual, expected)


def test_evaluate_initial_window():
    initial_window = 20
    y = make_forecasting_problem(n_timepoints=30, index_type="int")
    forecaster = NaiveForecaster()
    fh = 1
    cv = SlidingWindowSplitter(fh=fh, initial_window=initial_window)
    scoring = MeanAbsolutePercentageError(symmetric=True)
    out = evaluate(
        forecaster=forecaster, y=y, cv=cv, strategy="update", scoring=scoring
    )
    _check_evaluate_output(out, cv, y, scoring)
    assert out.loc[0, "len_train_window"] == initial_window

    # check scoring
    actual = out.loc[0, f"test_{scoring.name}"]
    train, test = next(cv.split(y))
    f = clone(forecaster)
    f.fit(y.iloc[train], fh=fh)
    expected = scoring(y.iloc[test], f.predict(), y_Train=y.iloc[train])
    np.testing.assert_equal(actual, expected)


def test_evaluate_no_exog_against_with_exog():
    # Check that adding exogenous data produces different results
    y, X = load_longley()
    forecaster = ARIMA(suppress_warnings=True)
    cv = SlidingWindowSplitter()
    scoring = MeanAbsolutePercentageError(symmetric=True)

    out_exog = evaluate(forecaster, cv, y, X=X, scoring=scoring)
    out_no_exog = evaluate(forecaster, cv, y, X=None, scoring=scoring)

    scoring_name = f"test_{scoring.name}"
    assert np.all(out_exog[scoring_name] != out_no_exog[scoring_name])
