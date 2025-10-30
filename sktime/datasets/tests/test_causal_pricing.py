#!/usr/bin/env python3
"""Tests for causal pricing data generator."""

__author__ = ["XAheli", "geetu040"]

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import make_causal_pricing
from sktime.datatypes import check_is_scitype
from sktime.utils.dependencies import _check_soft_dependencies

PGMPY_AVAILABLE = _check_soft_dependencies("pgmpy", severity="none")


class TestMakeCausalPricing:
    """Tests for make_causal_pricing function."""

    def test_output_shape(self):
        """Test output shape with custom dimensions."""
        n_series, n_timepoints = 10, 50
        X, y = make_causal_pricing(
            n_series=n_series,
            n_timepoints=n_timepoints,
            return_ground_truth=False,
            random_state=42,
        )

        assert X.shape == (n_series * n_timepoints, 7), "X shape incorrect"
        assert y.shape == (n_series * n_timepoints, 1), "y shape incorrect"

    def test_output_scitype(self):
        """Test output is Panel scitype."""
        X, y = make_causal_pricing(
            n_series=10, n_timepoints=20, return_ground_truth=False, random_state=42
        )

        assert check_is_scitype(X, "Panel"), "X is not Panel scitype"
        assert check_is_scitype(y, "Panel"), "y is not Panel scitype"

    def test_multiindex_structure(self):
        """Test MultiIndex structure."""
        n_series, n_timepoints = 5, 10
        X, y = make_causal_pricing(
            n_series=n_series,
            n_timepoints=n_timepoints,
            return_ground_truth=False,
            random_state=42,
        )

        assert isinstance(X.index, pd.MultiIndex), "X index is not MultiIndex"
        assert X.index.names == ["article_id", "time"], "Index names incorrect"
        assert X.index.get_level_values("article_id").nunique() == n_series
        assert X.index.get_level_values("time").nunique() == n_timepoints

        assert isinstance(y.index, pd.MultiIndex), "y index is not MultiIndex"
        assert y.index.names == ["article_id", "time"], "y index names incorrect"

    def test_column_names(self):
        """Test that X and y have correct columns."""
        X, y = make_causal_pricing(
            n_series=5, n_timepoints=10, random_state=42, return_ground_truth=False
        )

        expected_X_cols = [
            "discount",
            "stock",
            "week_number",
            "d",
            "k",
            "promotion",
            "p0",
        ]
        assert list(X.columns) == expected_X_cols, f"X columns incorrect: {X.columns}"

        assert list(y.columns) == ["demand"], f"y columns incorrect: {y.columns}"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_reproducibility(self):
        """Test random_state produces reproducible results."""
        X1, y1, gt1 = make_causal_pricing(n_series=10, n_timepoints=20, random_state=42)
        X2, y2, gt2 = make_causal_pricing(n_series=10, n_timepoints=20, random_state=42)

        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_frame_equal(y1, y2)
        pd.testing.assert_series_equal(
            gt1["treatment_effects"], gt2["treatment_effects"]
        )

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_different_random_states(self):
        """Test different random states produce different results."""
        X1, _, _ = make_causal_pricing(n_series=10, n_timepoints=20, random_state=42)
        X2, _, _ = make_causal_pricing(n_series=10, n_timepoints=20, random_state=123)

        assert not X1.equals(X2), (
            "Different random states should produce different data"
        )

    def test_physical_constraints_demand(self):
        """Test that demand is non-negative."""
        X, y = make_causal_pricing(
            n_series=50, n_timepoints=100, random_state=123, return_ground_truth=False
        )

        assert (y["demand"] >= 0).all(), "Found negative demand values"

    def test_physical_constraints_stock(self):
        """Test that stock is non-negative."""
        X, y = make_causal_pricing(
            n_series=50, n_timepoints=100, random_state=123, return_ground_truth=False
        )

        assert (X["stock"] >= 0).all(), "Found negative stock values"

    def test_physical_constraints_discount(self):
        """Test that discounts are in valid range [0, 0.5]."""
        X, y = make_causal_pricing(
            n_series=50, n_timepoints=100, random_state=123, return_ground_truth=False
        )

        assert (X["discount"] >= 0).all(), "Found negative discount values"
        assert (X["discount"] <= 0.5).all(), "Found discount values > 0.5"

    def test_physical_constraints_price(self):
        """Test that initial prices are positive."""
        X, y = make_causal_pricing(
            n_series=50, n_timepoints=100, random_state=123, return_ground_truth=False
        )

        assert (X["p0"] > 0).all(), "Found non-positive initial prices"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_treatment_effects_negative(self):
        """Test that treatment effects are negative (normal goods)."""
        _, _, gt = make_causal_pricing(n_series=100, n_timepoints=50, random_state=456)

        assert (gt["treatment_effects"] < 0).all(), (
            "Treatment effects should be negative"
        )

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_demand_price_relationship(self):
        """Test that higher price leads to lower demand on average."""
        X, y, gt = make_causal_pricing(n_series=100, n_timepoints=50, random_state=456)

        X["price"] = X["p0"] * (1 - X["discount"])
        merged = X.join(y)

        correlations = []
        for article_id in range(100):
            article_data = merged.loc[article_id]
            corr = article_data["price"].corr(article_data["demand"])
            correlations.append(corr)

        mean_corr = np.mean(correlations)
        assert mean_corr < 0, (
            f"Mean price-demand correlation should be negative, got {mean_corr}"
        )

    def test_return_ground_truth_false(self):
        """Test that return_ground_truth=False returns only X and y."""
        result = make_causal_pricing(
            n_series=5, n_timepoints=10, return_ground_truth=False, random_state=42
        )

        assert len(result) == 2, (
            "Should return only (X, y) when return_ground_truth=False"
        )
        X, y = result
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_return_ground_truth_true(self):
        """Test that return_ground_truth=True returns X, y, and ground_truth."""
        result = make_causal_pricing(
            n_series=5, n_timepoints=10, return_ground_truth=True, random_state=42
        )

        assert len(result) == 3, (
            "Should return (X, y, ground_truth) when return_ground_truth=True"
        )
        X, y, gt = result
        assert isinstance(gt, dict)
        assert "treatment_effects" in gt
        assert "base_demand" in gt
        assert "causal_dag" in gt
        assert "metadata" in gt

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_ground_truth_structure(self):
        """Test ground truth dictionary structure."""
        _, _, gt = make_causal_pricing(n_series=10, n_timepoints=20, random_state=42)

        assert isinstance(gt["treatment_effects"], pd.Series)
        assert gt["treatment_effects"].name == "treatment_effect"
        assert gt["treatment_effects"].index.name == "article_id"

        assert isinstance(gt["base_demand"], pd.DataFrame)
        assert list(gt["base_demand"].columns) == ["base_demand"]
        assert gt["base_demand"].index.names == ["article_id", "time"]

        assert "causal_dag" in gt
        assert "metadata" in gt
        assert isinstance(gt["metadata"], dict)

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_causal_dag_structure(self):
        """Test that causal DAG is properly structured."""

        from pgmpy.base import DAG

        _, _, gt = make_causal_pricing(n_series=5, n_timepoints=10, random_state=42)

        causal_dag = gt["causal_dag"]

        # Verify it's a pgmpy DAG object
        assert isinstance(causal_dag, DAG), (
            f"causal_dag should be pgmpy.base.DAG, got {type(causal_dag)}"
        )

        # Expected nodes and edges
        expected_nodes = [
            "base_demand",
            "price",
            "discount",
            "stock",
            "demand",
            "category_d",
            "category_k",
            "seasonality",
            "trend",
        ]

        expected_edges = [
            ("base_demand", "demand"),
            ("price", "demand"),
            ("stock", "discount"),
            ("discount", "price"),
            ("category_d", "base_demand"),
            ("category_k", "base_demand"),
            ("category_k", "seasonality"),
            ("seasonality", "base_demand"),
            ("trend", "base_demand"),
        ]

        # Test nodes
        dag_nodes = list(causal_dag.nodes())
        assert all(node in dag_nodes for node in expected_nodes), (
            "Missing nodes in causal DAG"
        )

        # Test edges
        for edge in expected_edges:
            assert causal_dag.has_edge(*edge), f"Missing edge {edge} in causal DAG"

        # Test key methods exist
        assert hasattr(causal_dag, "nodes"), "DAG missing nodes() method"
        assert hasattr(causal_dag, "edges"), "DAG missing edges() method"
        assert hasattr(causal_dag, "has_edge"), "DAG missing has_edge() method"
        assert hasattr(causal_dag, "get_parents"), "DAG missing get_parents() method"
        assert hasattr(causal_dag, "get_children"), "DAG missing get_children() method"

        # Test specific causal relationships
        assert causal_dag.has_edge("price", "demand"), "Missing price → demand edge"
        assert not causal_dag.has_edge("demand", "price"), (
            "Should not have demand → price"
        )

        # Test get_parents and get_children
        demand_parents = list(causal_dag.get_parents("demand"))
        assert "price" in demand_parents, "price should be parent of demand"
        assert "base_demand" in demand_parents, "base_demand should be parent of demand"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_metadata_content(self):
        """Test metadata contains expected information."""
        n_series, n_timepoints = 10, 20
        n_categories_d, n_categories_k = 45, 15
        random_state = 42

        _, _, gt = make_causal_pricing(
            n_series=n_series,
            n_timepoints=n_timepoints,
            n_categories_d=n_categories_d,
            n_categories_k=n_categories_k,
            random_state=random_state,
        )

        metadata = gt["metadata"]
        assert metadata["n_series"] == n_series
        assert metadata["n_timepoints"] == n_timepoints
        assert metadata["n_categories_d"] == n_categories_d
        assert metadata["n_categories_k"] == n_categories_k
        assert metadata["random_state"] == random_state
        assert "reference" in metadata

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_seasonal_pattern_exists(self):
        """Test that seasonality is present in the data."""
        _, _, gt = make_causal_pricing(n_series=10, n_timepoints=100, random_state=789)

        base_demand = gt["base_demand"].reset_index()
        article_0 = base_demand[base_demand["article_id"] == 0]["base_demand"].values

        from scipy.fft import fft, fftfreq

        n = len(article_0)
        freqs = fftfreq(n)
        fft_vals = np.abs(fft(article_0))

        positive_freqs = freqs[1 : n // 2]
        positive_fft = fft_vals[1 : n // 2]

        peak_idx = np.argmax(positive_fft)
        peak_freq = positive_freqs[peak_idx]
        peak_period = 1 / peak_freq

        assert 20 < peak_period < 40, (
            f"Expected seasonality period ~30, got {peak_period}"
        )

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_stock_depletes_over_time(self):
        """Test that stock generally decreases over time."""
        X, _, _ = make_causal_pricing(n_series=50, n_timepoints=100, random_state=321)

        initial_stock = X.xs(0, level="time")["stock"].mean()
        final_stock = X.xs(99, level="time")["stock"].mean()

        assert final_stock < initial_stock, "Stock should decrease over time on average"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_stock_clearing_behavior(self):
        """Test that stock approximately clears by end of season."""
        X, _, _ = make_causal_pricing(n_series=50, n_timepoints=100, random_state=321)

        initial_stock = X.xs(0, level="time")["stock"]
        final_stock = X.xs(99, level="time")["stock"]

        final_relative = (final_stock / initial_stock).mean()
        assert final_relative < 0.2, (
            f"Expected final stock < 20% of initial, got {final_relative * 100:.1f}%"
        )

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_discount_levels(self):
        """Test that discounts take discrete levels."""
        X, _, _ = make_causal_pricing(n_series=50, n_timepoints=100, random_state=123)

        unique_discounts = X["discount"].unique()
        expected_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        for discount in unique_discounts:
            assert any(
                np.isclose(discount, level, atol=1e-10) for level in expected_levels
            ), f"Unexpected discount level: {discount}"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_category_assignments(self):
        """Test that category assignments are within valid ranges."""
        X, _, _ = make_causal_pricing(
            n_series=100,
            n_timepoints=20,
            n_categories_d=45,
            n_categories_k=15,
            random_state=42,
        )

        assert (X["d"] >= 0).all() and (X["d"] < 45).all(), "Category d out of range"
        assert (X["k"] >= 0).all() and (X["k"] < 15).all(), "Category k out of range"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_promotion_binary(self):
        """Test that promotion is binary."""
        X, _, _ = make_causal_pricing(n_series=50, n_timepoints=20, random_state=42)

        unique_promotions = X["promotion"].unique()
        assert set(unique_promotions).issubset({0, 1}), (
            "Promotion should be binary (0 or 1)"
        )

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_week_number_sequential(self):
        """Test that week_number is sequential."""
        X, _, _ = make_causal_pricing(n_series=5, n_timepoints=20, random_state=42)

        for article_id in range(5):
            weeks = X.loc[article_id, "week_number"].values
            expected_weeks = np.arange(20)
            np.testing.assert_array_equal(weeks, expected_weeks)

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_average_discount_near_target(self):
        """Test that average discount is reasonable (not zero, not maxed out).

        Note: Paper targets ~14% average discount, but the pricing
        policy (Equation 29) allows variance based on stock levels and random
        discount adjustments.
        """
        X, _, _ = make_causal_pricing(n_series=100, n_timepoints=100, random_state=456)

        mean_discount = X["discount"].mean()
        assert 0.08 < mean_discount < 0.30, (
            f"Expected mean discount between 8% and 30%, got {mean_discount * 100:.1f}%"
        )

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_base_demand_positive(self):
        """Test that base demand is positive."""
        _, _, gt = make_causal_pricing(n_series=50, n_timepoints=50, random_state=123)

        base_demand = gt["base_demand"]["base_demand"]
        assert (base_demand > 0).all(), "Base demand should be positive"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_treatment_effect_heterogeneity(self):
        """Test that treatment effects show heterogeneity across articles."""
        _, _, gt = make_causal_pricing(n_series=100, n_timepoints=50, random_state=789)

        treatment_effects = gt["treatment_effects"]
        std_effects = treatment_effects.std()

        assert std_effects > 0, "Treatment effects should show heterogeneity"
        assert len(treatment_effects.unique()) > 50, (
            "Treatment effects should be diverse"
        )

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_custom_categories(self):
        """Test with custom number of categories."""
        n_categories_d, n_categories_k = 10, 5
        X, _, _ = make_causal_pricing(
            n_series=50,
            n_timepoints=20,
            n_categories_d=n_categories_d,
            n_categories_k=n_categories_k,
            random_state=42,
        )

        assert X["d"].max() < n_categories_d, "Category d exceeds n_categories_d"
        assert X["k"].max() < n_categories_k, "Category k exceeds n_categories_k"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_small_dataset(self):
        """Test generation with minimal dataset size."""
        X, y, gt = make_causal_pricing(n_series=2, n_timepoints=10, random_state=42)

        assert X.shape[0] == 20, "Small dataset shape incorrect"
        assert len(gt["treatment_effects"]) == 2

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_large_dataset(self):
        """Test generation with larger dataset (default paper size)."""
        X, y, gt = make_causal_pricing(n_series=100, n_timepoints=100, random_state=42)

        assert X.shape[0] == 10000
        assert y.shape[0] == 10000
        assert len(gt["treatment_effects"]) == 100

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    @pytest.mark.parametrize("random_state", [0, 42, 123, 999])
    def test_various_random_states(self, random_state):
        """Test that various random states work correctly."""
        X, y, gt = make_causal_pricing(
            n_series=10, n_timepoints=20, random_state=random_state
        )

        assert X.shape == (200, 7)
        assert (y["demand"] >= 0).all()
        assert (gt["treatment_effects"] < 0).all()

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_data_types(self):
        """Test that data types are appropriate."""
        X, y, gt = make_causal_pricing(n_series=10, n_timepoints=20, random_state=42)

        assert X["discount"].dtype in [np.float64, np.float32]
        assert X["stock"].dtype in [np.float64, np.float32]
        assert X["week_number"].dtype in [np.int64, np.int32]
        assert X["d"].dtype in [np.int64, np.int32]
        assert X["k"].dtype in [np.int64, np.int32]
        assert X["promotion"].dtype in [np.int64, np.int32]
        assert X["p0"].dtype in [np.float64, np.float32]

        assert y["demand"].dtype in [np.float64, np.float32]

    def test_no_missing_values(self):
        """Test that there are no missing values."""
        X, y = make_causal_pricing(
            n_series=50, n_timepoints=50, random_state=42, return_ground_truth=False
        )

        assert not X.isnull().any().any(), "Found missing values in X"
        assert not y.isnull().any().any(), "Found missing values in y"

    @pytest.mark.skipif(not PGMPY_AVAILABLE, reason="Requires pgmpy")
    def test_initial_price_stability(self):
        """Test that initial price (p0) is constant for each article."""
        X, _, _ = make_causal_pricing(n_series=10, n_timepoints=20, random_state=42)

        for article_id in range(10):
            article_p0 = X.loc[article_id, "p0"].values
            assert np.all(article_p0 == article_p0[0]), (
                f"p0 should be constant for article {article_id}"
            )
