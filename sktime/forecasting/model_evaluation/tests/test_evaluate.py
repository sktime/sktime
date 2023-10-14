#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for model evaluation module.

In particular, function `evaluate`, that performs time series cross-validation, is
tested with various configurations for correct output.
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
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.compose._reduce import DirectReductionForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_evaluation._functions import (
    _check_scores,
    _get_column_order_and_datatype,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.tests._config import TEST_FHS, TEST_STEP_LENGTHS_INT
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanAbsoluteScaledError,
)
from sktime.performance_metrics.forecasting.probabilistic import (
    CRPS,
    EmpiricalCoverage,
    LogLoss,
    PinballLoss,
)
from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.estimator_checks import _assert_array_almost_equal
from sktime.utils._testing.forecasting import make_forecasting_problem
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.utils._testing.series import _make_series
from sktime.utils.validation._dependencies import _check_soft_dependencies

METRICS = [MeanAbsolutePercentageError(symmetric=True), MeanAbsoluteScaledError()]
PROBA_METRICS = [CRPS(), EmpiricalCoverage(), LogLoss(), PinballLoss()]


def _check_evaluate_output(out, cv, y, scoring, return_data):
    assert isinstance(out, pd.DataFrame)
    # Check column names.
    scoring = _check_scores(scoring)
    columns = _get_column_order_and_datatype(scoring, return_data)
    assert set(out.columns) == columns.keys(), "Columns are not identical"

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


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
# Test using MAPE and MASE scorers so that tests cover a metric that doesn't
# use y_train (MAPE) and one that does use y_train (MASE).
@pytest.mark.parametrize("CV", [SlidingWindowSplitter, ExpandingWindowSplitter])
@pytest.mark.parametrize("fh", TEST_FHS)
@pytest.mark.parametrize("window_length", [7, 10])
@pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS_INT)
@pytest.mark.parametrize("strategy", ["refit", "update", "no-update_params"])
@pytest.mark.parametrize("scoring", METRICS)
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
    _check_evaluate_output(out, cv, y, scoring, False)

    # check scoring
    actual = out.loc[:, f"test_{scoring.name}"]

    n_splits = cv.get_n_splits(y)
    expected = np.empty(n_splits)
    for i, (train, test) in enumerate(cv.split(y)):
        f = forecaster.clone()
        f.fit(y.iloc[train], fh=fh)
        expected[i] = scoring(y.iloc[test], f.predict(), y_train=y.iloc[train])

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(
    not run_test_for_class([evaluate] + PROBA_METRICS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("return_data", [True, False])
@pytest.mark.parametrize("scores", [METRICS, PROBA_METRICS, METRICS + PROBA_METRICS])
def test_scoring_list(return_data, scores):
    y = make_forecasting_problem(n_timepoints=30, index_type="int")
    forecaster = NaiveForecaster()
    cv = SlidingWindowSplitter(fh=[1, 2, 3], initial_window=15, step_length=5)

    out = evaluate(
        forecaster=forecaster,
        y=y,
        cv=cv,
        scoring=scores,
        return_data=return_data,
    )
    _check_evaluate_output(out, cv, y, scores, return_data)


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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
    _check_evaluate_output(out, cv, y, scoring, False)
    assert out.loc[0, "len_train_window"] == initial_window

    # check scoring
    actual = out.loc[0, f"test_{scoring.name}"]
    train, test = next(cv.split(y))
    f = forecaster.clone()
    f.fit(y.iloc[train], fh=fh)
    expected = scoring(y.iloc[test], f.predict(), y_Train=y.iloc[train])
    np.testing.assert_equal(actual, expected)


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
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
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.skipif(
    not _check_soft_dependencies("statsmodels", severity="none"),
    reason="skip test if required soft dependency not available",
)
@pytest.mark.parametrize("error_score", [np.nan, "raise", 1000])
@pytest.mark.parametrize("return_data", [True, False])
@pytest.mark.parametrize("strategy", ["refit", "update", "no-update_params"])
@pytest.mark.parametrize("backend", [None, "dask", "loky", "threading"])
@pytest.mark.parametrize("scores", [[MeanAbsolutePercentageError()], METRICS])
def test_evaluate_error_score(error_score, return_data, strategy, backend, scores):
    """Test evaluate to raise warnings and exceptions according to error_score value."""
    # skip test for dask backend if dask is not installed
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        return None

    forecaster = ExponentialSmoothing(sp=12)
    y = load_airline()
    # add NaN to make ExponentialSmoothing fail
    y.iloc[1] = np.nan
    fh = [1, 2, 3]
    cv = SlidingWindowSplitter(step_length=33, initial_window=36, fh=fh)
    scoring_name = [f"test_{score.name}" for score in scores]

    args = {
        "forecaster": forecaster,
        "y": y,
        "cv": cv,
        "scoring": scores,
        "return_data": return_data,
        "error_score": error_score,
        "strategy": strategy,
        "backend": backend,
    }

    if error_score in [np.nan, 1000]:
        # known bug - loky backend does not pass on warnings, #5307
        if backend != "loky":
            with pytest.warns(FitFailedWarning):
                results = evaluate(**args)
        else:
            results = evaluate(**args)

        if isinstance(error_score, type(np.nan)):
            assert all(results[scoring_name].isna().sum() > 0)
        if error_score == 1000:
            assert all(results[scoring_name].max() == 1000)
    if error_score == "raise":
        with pytest.raises(Exception):  # noqa: B017
            evaluate(**args)


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("backend", [None, "dask", "loky", "threading"])
def test_evaluate_hierarchical(backend):
    """Check that evaluate works with hierarchical data."""
    # skip test for dask backend if dask is not installed
    if backend == "dask" and not _check_soft_dependencies("dask", severity="none"):
        return None

    y = _make_hierarchical(
        random_state=0, hierarchy_levels=(2, 2), min_timepoints=12, max_timepoints=12
    )
    X = _make_hierarchical(
        random_state=42, hierarchy_levels=(2, 2), min_timepoints=12, max_timepoints=12
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


# ARIMA models from statsmodels, pmdarima
ARIMA_MODELS = [ARIMA, AutoARIMA]

# breaks for SARIMAX, see issue #3670, this should be fixed
# ARIMA_MODELS = [ARIMA, AutoARIMA, SARIMAX]


@pytest.mark.parametrize("cls", ARIMA_MODELS)
def test_evaluate_bigger_X(cls):
    """Check that evaluating ARIMA models with exogeneous X works.

    Example adapted from bug report #3657.
    """
    if not run_test_for_class(cls):
        return None

    y, X = load_longley()

    f = cls.create_test_instance()
    cv = ExpandingWindowSplitter(initial_window=3, step_length=1, fh=np.arange(1, 4))
    loss = MeanAbsoluteError()

    # check that this does not break
    evaluate(forecaster=f, y=y, X=X, cv=cv, error_score="raise", scoring=loss)


@pytest.mark.skipif(
    not run_test_for_class([evaluate] + PROBA_METRICS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_columns", [1, 2])
@pytest.mark.parametrize("scoring", PROBA_METRICS)
def test_evaluate_probabilistic(n_columns, scoring):
    """Check that evaluate works with interval, quantile, and distribution forecasts."""
    y = _make_series(n_columns=n_columns)

    forecaster = NaiveForecaster()
    cv = SlidingWindowSplitter()
    try:
        out = evaluate(
            forecaster,
            cv,
            y,
            X=None,
            scoring=scoring,
            error_score="raise",
        )
        scoring_name = f"test_{scoring.name}"
        assert scoring_name in out.columns
    except NotImplementedError:
        pass


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_evaluate_hierarchical_unequal_X_y():
    """Test evaluate with hierarchical X and y where X is larger.

    Tests failure case in bug report #4842.
    """
    from sktime.transformations.hierarchical.aggregate import Aggregator

    # hierarchical/panel with 2-level pd.MultiIndex,
    # level 0 with "A", and "B", dates from 2020-01-01 to 2020-01-10
    df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [["A", "B"], pd.date_range("2020-01-01", "2020-01-10").to_period("D")]
        ),
        data={"target": np.arange(20) % 10},
    ).sort_index()
    df = Aggregator().fit_transform(df)

    y = df[df.index.get_level_values(-1) < "2020-01-08"]
    X = df.copy()
    cv = ExpandingWindowSplitter(initial_window=2, fh=[1], step_length=1)

    f = NaiveForecaster()

    # this fails in the case of #4842 as y and X have different length
    res = evaluate(f, cv, y, X, error_score="raise")

    # further sanity checks to pin down deterministic properties of return
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (5, 5)

    expected_cols = np.array([1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6])
    output_metrics = res.loc[:, "test_MeanAbsolutePercentageError"].values
    _assert_array_almost_equal(output_metrics, expected_cols)
