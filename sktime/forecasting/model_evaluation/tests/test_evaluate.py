#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for model evaluation module.

In particular, function `evaluate`, that performs time series
cross-validation, is tested with various configurations for correct output.
"""

__author__ = ["aiwalter", "mloning", "fkiraly"]
__all__ = [
    "test_evaluate_common_configs",
    "test_evaluate_initial_window",
    "test_evaluate_no_exog_against_with_exog",
]

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from sktime.datasets import load_airline, load_longley
from sktime.exceptions import FitFailedWarning
from sktime.forecasting.compose._reduce import DirectReductionForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    SlidingWindowSplitter,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.tests._config import TEST_FHS, TEST_STEP_LENGTHS_INT
from sktime.performance_metrics.forecasting import (
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
)
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils.validation._dependencies import _check_soft_dependencies


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
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS_INT)
@pytest.mark.parametrize("strategy", ["refit", "update"])
@pytest.mark.parametrize(
    "scoring",
    [
        MeanAbsolutePercentageError(symmetric=True),
        MeanAbsoluteScaledError(),
    ],
)
@pytest.mark.parametrize("backend", [None, "dask", "loky", "threading"])
def test_evaluate_common_configs(
    CV, fh, window_length, step_length, strategy, scoring, backend
):
    """Test evaluate common configs."""
    # skip test for dask backend if dask is not installed
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        return None

    y = make_forecasting_problem(n_timepoints=30, index_type="int")
    forecaster = NaiveForecaster()
    cv = CV(fh, window_length, step_length=step_length)

    out = evaluate(
        forecaster=forecaster,
        y=y,
        cv=cv,
        strategy=strategy,
        scoring=scoring,
        backend=backend,
    )
    _check_evaluate_output(out, cv, y, scoring)

    # check scoring
    actual = out.loc[:, f"test_{scoring.name}"]

    n_splits = cv.get_n_splits(y)
    expected = np.empty(n_splits)
    for i, (train, test) in enumerate(cv.split(y)):
        f = forecaster.clone()
        f.fit(y.iloc[train], fh=fh)
        expected[i] = scoring(y.iloc[test], f.predict(), y_train=y.iloc[train])

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("return_data", [True, False])
def test_scoring_list(return_data):
    y = make_forecasting_problem(n_timepoints=30, index_type="int")
    forecaster = NaiveForecaster()
    cv = SlidingWindowSplitter(fh=[1, 2, 3], initial_window=15, step_length=5)

    out = evaluate(
        forecaster=forecaster,
        y=y,
        cv=cv,
        scoring=[
            MeanAbsolutePercentageError(symmetric=True),
            MeanAbsoluteScaledError(),
        ],
        return_data=return_data,
    )
    assert "test_MeanAbsolutePercentageError" in out.columns
    assert "test_MeanAbsoluteScaledError" in out.columns
    if return_data:
        assert "y_pred" in out.columns
        assert "y_train" in out.columns
        assert "y_test" in out.columns
    else:
        assert "y_pred" not in out.columns
        assert "y_train" not in out.columns
        assert "y_test" not in out.columns


def test_evaluate_initial_window():
    """Test evaluate initial window."""
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
    f = forecaster.clone()
    f.fit(y.iloc[train], fh=fh)
    expected = scoring(y.iloc[test], f.predict(), y_Train=y.iloc[train])
    np.testing.assert_equal(actual, expected)


def test_evaluate_no_exog_against_with_exog():
    """Check that adding exogenous data produces different results."""
    y, X = load_longley()
    forecaster = DirectReductionForecaster(LinearRegression())
    cv = SlidingWindowSplitter()
    scoring = MeanAbsolutePercentageError(symmetric=True)

    out_exog = evaluate(forecaster, cv, y, X=X, scoring=scoring)
    out_no_exog = evaluate(forecaster, cv, y, X=None, scoring=scoring)

    scoring_name = f"test_{scoring.name}"
    assert np.all(out_exog[scoring_name] != out_no_exog[scoring_name])


@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("error_score", [np.nan, "raise", 1000])
@pytest.mark.parametrize("return_data", [True, False])
@pytest.mark.parametrize("strategy", ["refit", "update"])
@pytest.mark.parametrize("backend", [None, "dask", "loky", "threading"])
def test_evaluate_error_score(error_score, return_data, strategy, backend):
    """Test evaluate to raise warnings and exceptions according to error_score value."""
    # skip test for dask backend if dask is not installed
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        return None

    forecaster = ExponentialSmoothing(sp=12)
    y = load_airline()
    # add NaN to make ExponentialSmoothing fail
    y.iloc[1] = np.nan
    fh = [1, 2, 3]
    cv = ExpandingWindowSplitter(step_length=48, initial_window=12, fh=fh)
    if error_score in [np.nan, 1000]:
        with pytest.warns(FitFailedWarning):
            results = evaluate(
                forecaster=forecaster,
                y=y,
                cv=cv,
                return_data=return_data,
                error_score=error_score,
                strategy=strategy,
                backend=backend,
            )
        if isinstance(error_score, type(np.nan)):
            assert results["test_MeanAbsolutePercentageError"].isna().sum() > 0
        if error_score == 1000:
            assert results["test_MeanAbsolutePercentageError"].max() == 1000
    if error_score == "raise":
        with pytest.raises(Exception):
            evaluate(
                forecaster=forecaster,
                y=y,
                cv=cv,
                return_data=return_data,
                error_score=error_score,
                strategy=strategy,
            )


@pytest.mark.parametrize("backend", [None, "dask", "loky", "threading"])
def test_evaluate_hierarchical(backend):
    """Check that adding exogenous data produces different results."""
    # skip test for dask backend if dask is not installed
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        return None

    y = _make_hierarchical(
        random_state=0, hierarchy_levels=(2, 2), min_timepoints=20, max_timepoints=20
    )
    X = _make_hierarchical(
        random_state=42, hierarchy_levels=(2, 2), min_timepoints=20, max_timepoints=20
    )
    y = y.sort_index()
    X = X.sort_index()

    forecaster = DirectReductionForecaster(LinearRegression())
    cv = SlidingWindowSplitter()
    scoring = MeanAbsolutePercentageError(symmetric=True)
    out_exog = evaluate(
        forecaster, cv, y, X=X, scoring=scoring, error_score="raise", backend=backend
    )
    out_no_exog = evaluate(
        forecaster, cv, y, X=None, scoring=scoring, error_score="raise", backend=backend
    )

    scoring_name = f"test_{scoring.name}"
    assert np.all(out_exog[scoring_name] != out_no_exog[scoring_name])
