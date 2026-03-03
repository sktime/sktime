#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for `evaluate` function for TSR.

`evaluate` performs cross-validation for time series regression, including
sample-weight and multioutput support.
"""

__author__ = ["Omswastik-11"]

__all__ = ["TestEvaluate"]

import numpy as np
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
    """Tests for `evaluate` function using sktime components."""

    def test_evaluate_basic_functionality(self):
        """Test basic functionality of evaluate function with data."""
        X, y = make_regression_problem(random_state=0)
        n_splits = 3
        cv = KFold(n_splits=n_splits, shuffle=False)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=r2_score,
            error_score="raise",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_r2_score" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns

        assert all(result["fit_time"] >= 0)
        assert all(result["pred_time"] >= 0)

    def test_evaluate_with_multiple_metrics(self):
        """Test evaluate function with multiple scoring metrics."""
        X, y = make_regression_problem(random_state=1)
        cv = KFold(n_splits=3, shuffle=True, random_state=1)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=[mean_absolute_error, r2_score],
            error_score="raise",
        )

        assert "test_mean_absolute_error" in result.columns
        assert "test_r2_score" in result.columns

    def test_evaluate_with_sample_weight_and_return_data(self):
        """Test sample_weight routing and return_data payload."""
        X, y = make_regression_problem(n_instances=12, random_state=2)
        sample_weight = np.linspace(1.0, 2.0, num=len(y))
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=mean_squared_error,
            sample_weight=sample_weight,
            return_data=True,
            error_score="raise",
        )

        assert "sample_weight_train" in result.columns
        assert "sample_weight_test" in result.columns

        for i in range(len(result)):
            train_w = result["sample_weight_train"].iloc[i]
            test_w = result["sample_weight_test"].iloc[i]
            assert len(train_w) + len(test_w) == len(y)

    def test_evaluate_multioutput_support(self):
        """Test multioutput metric routing."""
        X, y = make_regression_problem(random_state=3)
        # create two-output target
        y_multi = pd.DataFrame({"a": y.squeeze(), "b": y.squeeze() + 1})
        cv = KFold(n_splits=2, shuffle=True, random_state=3)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y_multi,
            scoring=mean_squared_error,
            multioutput="uniform_average",
            error_score="raise",
        )

        assert "test_mean_squared_error" in result.columns
        assert len(result) == 2

    def test_evaluate_with_different_regressors(self):
        """Test evaluate function with different regressor types."""
        X, y = make_regression_problem(random_state=4)
        cv = KFold(n_splits=2, shuffle=False)

        regressors = [DummyRegressor(), KNeighborsTimeSeriesRegressor(n_neighbors=1)]

        for regressor in regressors:
            result = evaluate(
                regressor=regressor,
                cv=cv,
                X=X,
                y=y,
                scoring=r2_score,
                error_score="raise",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "test_r2_score" in result.columns

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_evaluate_parallel_backend(self, backend):
        """Test the parallelization backends."""
        X, y = make_regression_problem(random_state=5)
        n_splits = 3
        cv = KFold(n_splits=n_splits)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=r2_score,
            error_score="raise",
            **backend,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_r2_score" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns

    def test_evaluate_parallel_backend_none(self):
        """Test the sequential loop if `backend="None"`."""
        X, y = make_regression_problem(random_state=6)
        n_splits = 3
        cv = KFold(n_splits=n_splits)

        result = evaluate(
            regressor=DummyRegressor(),
            cv=cv,
            X=X,
            y=y,
            scoring=r2_score,
            error_score="raise",
            backend="None",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits
        assert "test_r2_score" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns
