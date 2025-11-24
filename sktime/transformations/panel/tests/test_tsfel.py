"""TSFEL test code."""

import pytest

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.tsfel import TSFELTransformer


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfel_on_basic_motions():
    """Test of TSFELTransformer on basic motions data."""
    X_train, _ = load_basic_motions(split="train")
    
    transformer = TSFELTransformer(features="minimal")
    result = transformer.fit_transform(X_train.iloc[:5])
    
    assert result.shape[0] == 5
    assert result.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFELTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfel_all_features():
    """Test TSFELTransformer with all features."""
    X_train, _ = load_basic_motions(split="train")
    
    transformer = TSFELTransformer(features="all")
    result = transformer.fit_transform(X_train.iloc[:3])
    
    assert result.shape[0] == 3
    assert result.shape[1] > 0

