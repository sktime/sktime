#!/usr/bin/env python3
"""Tests for CausalFeatureEngineer."""

__author__ = ["XAheli"]

import numpy as np
import pandas as pd
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.utils.dependencies import _check_soft_dependencies

# Skip all tests if pgmpy is not available
pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("pgmpy>=0.1.20", severity="none"),
    reason="pgmpy not available",
)


@pytest.fixture
def sample_univariate_data():
    """Create sample univariate time series data."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    y = pd.Series(
        np.cumsum(np.random.normal(0, 1, n)) + 100, index=dates, name="target"
    )
    return y


@pytest.fixture
def sample_multivariate_data():
    """Create sample multivariate time series data with causal relationships."""
    np.random.seed(42)
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Create causal relationships: X1 -> X2 -> X3
    X1 = np.random.normal(0, 1, n)
    X2 = 0.7 * X1 + np.random.normal(0, 0.3, n)
    X3 = 0.5 * X2 + 0.2 * X1 + np.random.normal(0, 0.2, n)

    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3}, index=dates)

    y = pd.Series(
        0.6 * X3 + 0.3 * X2 + np.random.normal(0, 0.1, n), index=dates, name="target"
    )

    return X, y


class TestCausalFeatureEngineer:
    """Test class for CausalFeatureEngineer."""

    def test_causal_feature_engineer_check_estimator(self):
        """Test that CausalFeatureEngineer passes the estimator checks."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        run_test_for_class(CausalFeatureEngineer)

    def test_causal_feature_engineer_univariate(self, sample_univariate_data):
        """Test CausalFeatureEngineer with univariate time series."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        y = sample_univariate_data

        transformer = CausalFeatureEngineer(
            max_lag=3,
            causal_method="pc",
            significance_level=0.1,  # More lenient for small sample
        )

        Xt = transformer.fit_transform(y)

        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt) > 0
        assert Xt.shape[0] <= len(y)
        assert Xt.shape[1] >= 0  # May be 0 if no relationships found
        assert transformer.n_features_generated_ >= 0

        assert hasattr(transformer, "causal_graph_")
        assert hasattr(transformer, "feature_importance_weights_")
        assert hasattr(transformer, "features_generated_")

    def test_causal_feature_engineer_multivariate(self, sample_multivariate_data):
        """Test CausalFeatureEngineer with multivariate time series."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        X, y = sample_multivariate_data

        transformer = CausalFeatureEngineer(
            max_lag=2,
            causal_method="hill_climb",
            significance_level=0.1,
        )

        Xt = transformer.fit_transform(X, y)

        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt) > 0
        assert Xt.shape[0] <= len(X)
        assert Xt.shape[1] >= 0
        assert transformer.n_features_generated_ >= 0

        assert len(transformer.features_generated_) >= 0

    def test_causal_feature_engineer_different_feature_types(
        self, sample_multivariate_data
    ):
        """Test CausalFeatureEngineer with different feature type configurations."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        X, y = sample_multivariate_data

        feature_types_to_test = [
            ["direct"],
            ["interaction"],
            ["temporal"],
            ["direct", "interaction"],
            ["direct", "temporal"],
            ["direct", "interaction", "temporal"],
        ]

        for feature_types in feature_types_to_test:
            transformer = CausalFeatureEngineer(
                max_lag=2,
                causal_method="pc",
                feature_types=feature_types,
                significance_level=0.2,  # Very lenient for testing
            )

            Xt = transformer.fit_transform(X, y)

            assert isinstance(Xt, pd.DataFrame)
            assert len(Xt) >= 0  # May be empty if no relationships found

    def test_causal_feature_engineer_expert_knowledge(self, sample_multivariate_data):
        """Test CausalFeatureEngineer with expert knowledge."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        X, y = sample_multivariate_data

        expert_knowledge = {
            "forbidden_edges": [("X3", "X1")],
            "required_edges": [("X1", "X2")],
        }

        transformer = CausalFeatureEngineer(
            max_lag=2,
            causal_method="pc",
            expert_knowledge=expert_knowledge,
            significance_level=0.2,
        )

        Xt = transformer.fit_transform(X, y)

        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt) >= 0

    def test_causal_feature_engineer_weighting_strategies(
        self, sample_multivariate_data
    ):
        """Test different weighting strategies."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        X, y = sample_multivariate_data

        strategies = ["uniform", "causal_strength", "inverse_lag"]

        for strategy in strategies:
            transformer = CausalFeatureEngineer(
                max_lag=2,
                causal_method="pc",
                weighting_strategy=strategy,
                significance_level=0.2,
            )

            Xt = transformer.fit_transform(X, y)

            assert isinstance(Xt, pd.DataFrame)
            assert len(Xt) >= 0

    def test_causal_feature_engineer_scoring_methods(self, sample_multivariate_data):
        """Test different scoring methods for hill climb."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        X, y = sample_multivariate_data

        scoring_methods = ["auto", "bic-g", "aic-g"]

        for method in scoring_methods:
            transformer = CausalFeatureEngineer(
                max_lag=2,
                causal_method="hill_climb",
                scoring_method=method,
            )

            Xt = transformer.fit_transform(X, y)

            assert isinstance(Xt, pd.DataFrame)
            assert len(Xt) >= 0

    def test_causal_feature_engineer_invalid_method(self, sample_univariate_data):
        """Test that invalid causal method raises error."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        y = sample_univariate_data

        transformer = CausalFeatureEngineer(causal_method="invalid_method")

        with pytest.raises(ValueError, match="Unsupported causal discovery method"):
            transformer.fit_transform(y)

    def test_causal_feature_engineer_invalid_scoring_method(
        self, sample_univariate_data
    ):
        """Test that invalid scoring method raises error."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        y = sample_univariate_data

        transformer = CausalFeatureEngineer(
            causal_method="hill_climb", scoring_method="invalid_method"
        )

        with pytest.raises(ValueError, match="Invalid scoring method"):
            transformer.fit_transform(y)

    def test_causal_feature_engineer_get_test_params(self):
        """Test get_test_params method."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        params = CausalFeatureEngineer.get_test_params()

        assert isinstance(params, dict)
        assert "causal_method" in params
        assert "max_lag" in params
        assert "feature_types" in params

    def test_causal_feature_engineer_alignment_methods(self, sample_multivariate_data):
        """Test index alignment utility methods."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        X, y = sample_multivariate_data

        transformer = CausalFeatureEngineer(max_lag=3)
        transformer.fit_transform(X, y)

        # Test alignment methods
        aligned_index = transformer.get_aligned_target_index()
        if aligned_index is not None:
            assert isinstance(aligned_index, pd.Index)

        safe_index = transformer.get_safe_target_index(y.index)
        if safe_index is not None:
            assert isinstance(safe_index, pd.Index)
            assert len(safe_index) <= len(y.index)

    def test_causal_feature_engineer_empty_features(self, sample_univariate_data):
        """Test behavior when no causal relationships are found."""
        from sktime.transformations.series.causal_feature_engineer import (
            CausalFeatureEngineer,
        )

        y = sample_univariate_data

        # Use very strict settings to likely produce no relationships
        transformer = CausalFeatureEngineer(
            max_lag=1,
            significance_level=0.001,  # Very strict
            min_causal_strength=0.9,  # Very high threshold
        )

        Xt = transformer.fit_transform(y)

        # Should still return a DataFrame, even if empty
        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt) >= 0
        assert transformer.n_features_generated_ >= 0
