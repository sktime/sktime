#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for CausalFeatureEngineer."""

__author__ = ["XAheli"]

import pytest
import pandas as pd
import numpy as np

from sktime.transformations.series.causal_feature_engineer import CausalFeatureEngineer
from sktime.datasets import load_airline, load_longley
from sktime.tests.test_switch import run_test_for_class

# Skip tests if pgmpy is not available
try:
    from pgmpy.estimators import PC
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False


@pytest.mark.skipif(not PGMPY_AVAILABLE, reason="pgmpy not available")
def test_causal_feature_engineer_check_estimator():
    """Test that CausalFeatureEngineer passes the estimator checks."""
    run_test_for_class(CausalFeatureEngineer)


@pytest.mark.skipif(not PGMPY_AVAILABLE, reason="pgmpy not available")
def test_causal_feature_engineer_univariate():
    """Test CausalFeatureEngineer with univariate time series."""
    y = load_airline()
    
    transformer = CausalFeatureEngineer(max_lag=3, causal_method="pc")
    
    Xt = transformer.fit_transform(y)
    
    assert isinstance(Xt, pd.DataFrame)
    assert len(Xt) > 0
    assert Xt.shape[0] <= len(y)
    assert Xt.shape[1] >= 1
    assert transformer.n_features_generated_ >= 1
    
    assert hasattr(transformer, "causal_graph_")
    assert hasattr(transformer, "feature_importance_weights_")
    assert hasattr(transformer, "features_generated_")


@pytest.mark.skipif(not PGMPY_AVAILABLE, reason="pgmpy not available")
def test_causal_feature_engineer_multivariate():
    """Test CausalFeatureEngineer with multivariate time series."""
    y, X = load_longley()
    
    transformer = CausalFeatureEngineer(max_lag=2, causal_method="hill_climb")
    
    Xt = transformer.fit_transform(X, y)
    
    assert isinstance(Xt, pd.DataFrame)
    assert len(Xt) > 0
    assert Xt.shape[0] <= len(X)
    assert Xt.shape[1] >= 1
    assert transformer.n_features_generated_ >= 1
    
    assert len(transformer.features_generated_) >= 1


@pytest.mark.skipif(not PGMPY_AVAILABLE, reason="pgmpy not available")
def test_causal_feature_engineer_different_feature_types():
    """Test CausalFeatureEngineer with different feature type configurations."""
    np.random.seed(42)
    n = 100
    
    X1 = pd.Series(np.random.normal(0, 1, n), name="X1")
    X2 = pd.Series(0.7 * X1 + np.random.normal(0, 0.3, n), name="X2")
    X3 = pd.Series(0.3 * X1 + 0.5 * X2 + np.random.normal(0, 0.2, n), name="X3")
    
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    
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
            feature_types=feature_types
        )
        
        Xt = transformer.fit_transform(X)
        
        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt) > 0


@pytest.mark.skipif(not PGMPY_AVAILABLE, reason="pgmpy not available")
def test_causal_feature_engineer_expert_knowledge():
    """Test CausalFeatureEngineer with expert knowledge."""
    np.random.seed(42)
    n = 100
    
    X1 = pd.Series(np.random.normal(0, 1, n), name="X1")
    X2 = pd.Series(0.7 * X1 + np.random.normal(0, 0.3, n), name="X2")
    X3 = pd.Series(0.3 * X1 + 0.5 * X2 + np.random.normal(0, 0.2, n), name="X3")
    
    X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    
    expert_knowledge = {
        "forbidden_edges": [("X3", "X1")],
        "required_edges": [("X1", "X2")],
    }
    
    transformer = CausalFeatureEngineer(
        max_lag=2,
        causal_method="pc",
        expert_knowledge=expert_knowledge
    )
    
    Xt = transformer.fit_transform(X)
    
    assert isinstance(Xt, pd.DataFrame)
    assert len(Xt) > 0
