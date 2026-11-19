#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for `evaluate` function for time series clustering.

`evaluate` performs cross-validation for time series clustering,
here it is tested with various configurations.
"""

__author__ = ["Nischal1425"]

__all__ = ["TestClusteringEvaluate"]

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import KFold

from sktime.clustering.base import BaseClusterer
from sktime.clustering.model_evaluation import evaluate
from sktime.tests.test_switch import run_test_for_class
from sktime.utils._testing.panel import make_clustering_problem
from sktime.utils.parallel import _get_parallel_test_fixtures

BACKENDS = _get_parallel_test_fixtures("estimator")


class _DummyClusterer(BaseClusterer):
    """Minimal clusterer for testing. Assigns deterministic cluster labels."""

    _tags = {
        "capability:out_of_sample": True,
        "capability:predict": True,
    }

    def __init__(self, n_clusters=3, random_state=42):
        self.random_state = random_state
        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        self.labels_ = rng.randint(0, self.n_clusters, size=X.shape[0])
        return self

    def _predict(self, X, y=None):
        rng = np.random.RandomState(self.random_state)
        return rng.randint(0, self.n_clusters, size=X.shape[0])


class _FailingClusterer(BaseClusterer):
    """Clusterer that always raises an exception during fit."""

    _tags = {
        "capability:out_of_sample": True,
        "capability:predict": True,
    }

    def __init__(self, n_clusters=3):
        super().__init__(n_clusters=n_clusters)

    def _fit(self, X, y=None):
        raise RuntimeError("Intentional failure for testing")

    def _predict(self, X, y=None):
        raise RuntimeError("Intentional failure for testing")


@pytest.mark.skipif(
    not run_test_for_class(evaluate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
class TestClusteringEvaluate:
    """Tests for `evaluate` function using sktime clustering components."""

    def test_evaluate_basic_functionality(self):
        """Test basic functionality with silhouette_score."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        n_splits = 3
        cv = KFold(n_splits=n_splits, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            scoring=silhouette_score,
            error_score="raise",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits

        assert "test_silhouette_score" in result.columns
        assert "fit_time" in result.columns
        assert "internal_time" in result.columns

        assert all(result["fit_time"] >= 0)

    def test_evaluate_multiple_internal_metrics(self):
        """Test evaluate with multiple internal metrics."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            scoring=[silhouette_score, calinski_harabasz_score],
            error_score="raise",
        )

        assert "test_silhouette_score" in result.columns
        assert "test_calinski_harabasz_score" in result.columns

    def test_evaluate_return_data(self):
        """Test evaluate function with return_data=True."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=2, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            scoring=silhouette_score,
            return_data=True,
            error_score="raise",
        )

        expected_data_cols = ["X_train", "X_test", "y_train", "y_test"]
        for col in expected_data_cols:
            assert col in result.columns

        for i in range(len(result)):
            assert result["X_train"].iloc[i] is not None
            assert result["X_test"].iloc[i] is not None

    def test_evaluate_different_cv_splits(self):
        """Test evaluate with different numbers of CV splits."""
        X = make_clustering_problem(n_instances=20, random_state=42)

        for n_splits in [2, 3, 5]:
            cv = KFold(n_splits=n_splits, shuffle=False)

            result = evaluate(
                clusterer=_DummyClusterer(n_clusters=3),
                cv=cv,
                X=X,
                scoring=silhouette_score,
            )

            assert len(result) == n_splits

    def test_evaluate_default_scoring(self):
        """Test that default metric is silhouette_score."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
        )

        assert "test_silhouette_score" in result.columns

    def test_evaluate_default_cv(self):
        """Test that default CV is KFold(n_splits=3)."""
        X = make_clustering_problem(n_instances=20, random_state=42)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            X=X,
        )

        assert len(result) == 3

    def test_evaluate_int_cv(self):
        """Test passing cv as integer."""
        X = make_clustering_problem(n_instances=20, random_state=42)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=5,
            X=X,
        )

        assert len(result) == 5

    def test_evaluate_timing(self):
        """Test that timing measurements are non-negative."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=2, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            scoring=silhouette_score,
            error_score="raise",
        )

        assert all(result["fit_time"] >= 0)
        assert all(result["internal_time"] >= 0)

    def test_evaluate_scores_are_numeric(self):
        """Test all score columns are numeric dtype."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            scoring=[silhouette_score, calinski_harabasz_score],
            error_score="raise",
        )

        for col in ["test_silhouette_score", "test_calinski_harabasz_score"]:
            assert pd.api.types.is_numeric_dtype(result[col]), (
                f"Column {col} is not numeric"
            )

    def test_evaluate_external_metric_with_labels(self):
        """Test external metric (adjusted_rand_score) with ground truth."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        # Create fake ground truth labels
        y = pd.DataFrame(
            np.random.RandomState(42).randint(0, 3, size=20), columns=["label"]
        )
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            y=y,
            scoring=adjusted_rand_score,
            error_score="raise",
        )

        assert "test_adjusted_rand_score" in result.columns
        assert len(result) == 3

    def test_evaluate_external_metric_without_labels(self):
        """Test external metric raises ValueError when y=None."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=3, shuffle=False)

        with pytest.raises(ValueError, match="External metrics.*require ground truth"):
            evaluate(
                clusterer=_DummyClusterer(n_clusters=3),
                cv=cv,
                X=X,
                y=None,
                scoring=adjusted_rand_score,
                error_score="raise",
            )

    def test_evaluate_mixed_metrics(self):
        """Test using both internal and external metrics together."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        y = pd.DataFrame(
            np.random.RandomState(42).randint(0, 3, size=20), columns=["label"]
        )
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            y=y,
            scoring=[silhouette_score, adjusted_rand_score],
            error_score="raise",
        )

        assert "test_silhouette_score" in result.columns
        assert "test_adjusted_rand_score" in result.columns

    def test_evaluate_error_score_numeric(self):
        """Test error_score=np.nan with a failing clusterer."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=3, shuffle=False)

        with pytest.warns(match="fitting of clusterer"):
            result = evaluate(
                clusterer=_FailingClusterer(n_clusters=3),
                cv=cv,
                X=X,
                scoring=silhouette_score,
                error_score=np.nan,
            )

        assert len(result) == 3
        assert all(np.isnan(result["test_silhouette_score"]))

    def test_evaluate_invalid_clusterer_type(self):
        """Test that non-BaseClusterer raises TypeError."""
        X = make_clustering_problem(n_instances=20, random_state=42)

        with pytest.raises(TypeError, match="Expected clusterer.*BaseClusterer"):
            evaluate(
                clusterer="not_a_clusterer",
                X=X,
            )

    def test_evaluate_davies_bouldin_metric(self):
        """Test Davies-Bouldin index metric works correctly."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            scoring=davies_bouldin_score,
            error_score="raise",
        )

        assert "test_davies_bouldin_score" in result.columns
        # Davies-Bouldin index is non-negative
        assert all(result["test_davies_bouldin_score"] >= 0)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_evaluate_parallel_backend(self, backend):
        """Test parallelization backends."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        n_splits = 3
        cv = KFold(n_splits=n_splits)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            scoring=silhouette_score,
            error_score="raise",
            **backend,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_splits
        assert "test_silhouette_score" in result.columns

    def test_evaluate_unsupervised_no_y(self):
        """Test evaluate works without y for internal metrics."""
        X = make_clustering_problem(n_instances=20, random_state=42)
        cv = KFold(n_splits=3, shuffle=False)

        result = evaluate(
            clusterer=_DummyClusterer(n_clusters=3),
            cv=cv,
            X=X,
            y=None,
            scoring=silhouette_score,
            error_score="raise",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "test_silhouette_score" in result.columns
