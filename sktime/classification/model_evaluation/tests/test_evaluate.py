#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for `evaluate` function for TSC.

`evaluate` performs cross-validation for time series
classification, here it is tested with various configurations.
"""

__author__ = ["jgyasu", "ksharma6"]

__all__ = ["TestEvaluate"]

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import KFold

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.dummy import DummyClassifier
from sktime.classification.model_evaluation import evaluate
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_classification_problem


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
class TestEvaluate:
    """Tests for `evaluate` function using sktime components."""

    def test_evaluate_basic_functionality(self):
        """Test basic functionality of evaluate function with  data."""
        X, y = make_classification_problem()
        n_splits = 3
        cv = KFold(n_splits=n_splits, shuffle=False)

        result = evaluate(
            classifier=DummyClassifier(),
            cv=cv,
            X=X,
            y=y,
            scoring=accuracy_score,
            error_score="raise",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_accuracy_score" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns

        assert all(result["test_accuracy_score"].between(0, 1))
        assert all(result["fit_time"] >= 0)
        assert all(result["pred_time"] >= 0)

    def test_evaluate_with_multiple_metrics(self):
        """Test evaluate function with multiple scoring metrics."""
        X, y = make_classification_problem()
        cv = KFold(n_splits=3, shuffle=True)

        result = evaluate(
            classifier=DummyClassifier(),
            cv=cv,
            X=X,
            y=y,
            scoring=[accuracy_score, f1_score],
            error_score="raise",
        )

        assert "test_accuracy_score" in result.columns
        assert "test_f1_score" in result.columns

        assert all(result["test_accuracy_score"].between(0, 1))
        assert all(result["test_f1_score"].between(0, 1))

    def test_evaluate_with_probabilistic_metrics(self):
        """Test evaluate function with probabilistic scoring metrics."""
        X, y = make_classification_problem()
        cv = KFold(n_splits=3, shuffle=True)

        result = evaluate(
            classifier=DummyClassifier(),
            cv=cv,
            X=X,
            y=y,
            scoring=[brier_score_loss],
            error_score="raise",
        )

        assert "test_brier_score_loss" in result.columns

        assert all(result["test_brier_score_loss"].between(0, 1))

    def test_evaluate_with_return_data(self):
        """Test evaluate function with return_data=True."""
        X, y = make_classification_problem()
        cv = KFold(n_splits=2, shuffle=False)

        result = evaluate(
            classifier=DummyClassifier(),
            cv=cv,
            X=X,
            y=y,
            scoring=accuracy_score,
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
        X, y = make_classification_problem()

        for n_splits in [2, 3, 5]:
            cv = KFold(n_splits=n_splits, shuffle=False)

            result = evaluate(
                classifier=DummyClassifier(), cv=cv, X=X, y=y, scoring=accuracy_score
            )

            assert len(result) == n_splits
            assert all(
                col in result.columns
                for col in ["test_accuracy_score", "fit_time", "pred_time"]
            )

    def test_evaluate_with_different_classifiers(self):
        """Test evaluate function with different types of classifiers."""
        X, y = make_classification_problem()
        cv = KFold(n_splits=2, shuffle=False)

        classifiers = [DummyClassifier(), KNeighborsTimeSeriesClassifier()]

        for classifier in classifiers:
            result = evaluate(
                classifier=classifier,
                cv=cv,
                X=X,
                y=y,
                scoring=accuracy_score,
                error_score="raise",
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "test_accuracy_score" in result.columns

    def test_evaluate_timing_measurements(self):
        """Test that evaluate function properly measures timing."""
        X, y = make_classification_problem()
        cv = KFold(n_splits=2, shuffle=False)

        result = evaluate(
            classifier=DummyClassifier(),
            cv=cv,
            X=X,
            y=y,
            scoring=accuracy_score,
            error_score="raise",
        )

        assert all(result["fit_time"] > 0)
        assert all(result["pred_time"] >= 0)

    def test_evaluate_parallel_backend(self):
        """Test the parrelelization backends"""
        X, y = make_classification_problem()
        n_splits = 3
        cv = KFold(n_splits=n_splits)

        result = evaluate(
            classifier=DummyClassifier(),
            cv=cv,
            X=X,
            y=y,
            scoring=accuracy_score,
            error_score="raise",
            backend="loky",
            backend_params={"n_jobs": -1},
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_accuracy_score" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns

    def test_evaluate_parallel_backend_none(self):
        """Test the sequential loop if `backend="None"`"""
        X, y = make_classification_problem()
        n_splits = 3
        cv = KFold(n_splits=n_splits)

        result = evaluate(
            classifier=DummyClassifier(),
            cv=cv,
            X=X,
            y=y,
            scoring=accuracy_score,
            error_score="raise",
            backend="None",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_accuracy_score" in result.columns
        assert "fit_time" in result.columns
        assert "pred_time" in result.columns
