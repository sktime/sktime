#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for `evaluate` function for time series regression.

`evaluate` performs cross-validation for time series
regression, here it is tested with various configurations.
"""

__author__ = ["NAME-ASHWANIYADAV"]

__all__ = ["TestEvaluate"]

import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.dummy import DummyRegressor
from sktime.regression.model_evaluation import evaluate
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_regression_problem
from sktime.utils.parallel import _get_parallel_test_fixtures

BACKENDS = _get_parallel_test_fixtures("estimator")


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
class TestEvaluate:
    """Tests for `evaluate` function using sktime regression components."""

    def test_evaluate_basic_functionality(self):
        """Test basic functionality of evaluate function with regression data."""
        X, y = make_regression_problem()
        n_splits = 3
        cv = KFold(n_splits=n_splits, shuffle=False)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=mean_squared_error,
            error_score="raise",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_mean_squared_error" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns

        assert all(result["test_mean_squared_error"] >= 0)
        assert all(result["fit_time"] >= 0)
        assert all(result["pred_time"] >= 0)

    def test_evaluate_with_multiple_metrics(self):
        """Test evaluate function with multiple scoring metrics."""
        X, y = make_regression_problem()
        cv = KFold(n_splits=3, shuffle=True)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=[mean_squared_error, mean_absolute_error],
            error_score="raise",
        )

        assert "test_mean_squared_error" in result.columns
        assert "test_mean_absolute_error" in result.columns

        assert all(result["test_mean_squared_error"] >= 0)
        assert all(result["test_mean_absolute_error"] >= 0)

    def test_evaluate_with_r2_score(self):
        """Test evaluate function with R2 score metric."""
        X, y = make_regression_problem()
        cv = KFold(n_splits=3, shuffle=True)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=[r2_score],
            error_score="raise",
        )

        assert "test_r2_score" in result.columns
        # R2 can be negative for bad models, so just check it's finite
        assert all(result["test_r2_score"].notna())

    def test_evaluate_with_return_data(self):
        """Test evaluate function with return_data=True."""
        X, y = make_regression_problem()
        cv = KFold(n_splits=2, shuffle=False)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=mean_squared_error,
            return_data=True,
            error_score="raise",
        )

        expected_data_cols = ["X_train", "X_test", "y_train", "y_test", "y_pred"]
        for col in expected_data_cols:
            assert col in result.columns

        for i in range(len(result)):
            assert result["X_train"].iloc[i] is not None
            assert result["X_test"].iloc[i] is not None
            assert result["y_train"].iloc[i] is not None
            assert result["y_test"].iloc[i] is not None

    def test_evaluate_different_cv_splits(self):
        """Test evaluate function with different numbers of CV splits."""
        X, y = make_regression_problem()

        for n_splits in [2, 3, 5]:
            cv = KFold(n_splits=n_splits, shuffle=False)

            result = evaluate(
                regressor=DummyRegressor(),
                cv=cv,
                X=X,
                y=y,
                scoring=mean_squared_error,
            )

            assert len(result) == n_splits
            assert all(
                col in result.columns
                for col in ["test_mean_squared_error", "fit_time", "pred_time"]
            )

    def test_evaluate_with_different_regressors(self):
        """Test evaluate function with different types of regressors."""
        X, y = make_regression_problem()
        cv = KFold(n_splits=2, shuffle=False)

        regressors = [DummyRegressor(), KNeighborsTimeSeriesRegressor()]

        for regressor in regressors:
            result = evaluate(
                regressor=regressor,
                cv=cv,
                X=X,
                y=y,
                scoring=mean_squared_error,
                error_score="raise",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "test_mean_squared_error" in result.columns

    def test_evaluate_timing_measurements(self):
        """Test that evaluate function properly measures timing."""
        X, y = make_regression_problem()
        cv = KFold(n_splits=2, shuffle=False)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=mean_squared_error,
            error_score="raise",
        )

        assert all(result["fit_time"] > 0)
        assert all(result["pred_time"] >= 0)

    def test_evaluate_default_scoring(self):
        """Test evaluate function with default scoring (None -> MSE)."""
        X, y = make_regression_problem()
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            error_score="raise",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "test_mean_squared_error" in result.columns

    def test_evaluate_default_cv(self):
        """Test evaluate function with default cv (None -> 3-fold)."""
        X, y = make_regression_problem()

        result = evaluate(
            regressor=DummyRegressor(),
            X=X,
            y=y,
            scoring=mean_squared_error,
            error_score="raise",
        )

        assert isinstance(result, pd.DataFrame)
        # default is 3-fold
        assert len(result) == 3

    def test_evaluate_int_cv(self):
        """Test evaluate function with integer cv parameter."""
        X, y = make_regression_problem()

        result = evaluate(
            regressor=DummyRegressor(),
            cv=5,
            X=X,
            y=y,
            scoring=mean_squared_error,
            error_score="raise",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_evaluate_parallel_backend(self, backend):
        """Test the parallelization backends."""
        X, y = make_regression_problem()
        n_splits = 3
        cv = KFold(n_splits=n_splits)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=mean_squared_error,
            error_score="raise",
            **backend,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_mean_squared_error" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns

    def test_evaluate_parallel_backend_none(self):
        """Test the sequential loop if `backend="None"`."""
        X, y = make_regression_problem()
        n_splits = 3
        cv = KFold(n_splits=n_splits)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=mean_squared_error,
            error_score="raise",
            backend="None",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_mean_squared_error" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns
