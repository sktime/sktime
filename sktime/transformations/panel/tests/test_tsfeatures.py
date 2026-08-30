"""TSFeatures test code."""

import pytest

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.tsfeatures import TSFeaturesTransformer


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_on_basic_motions():
    """Test of TSFeaturesTransformer on basic motions data."""
    X_train, _ = load_basic_motions(split="train")
    
    transformer = TSFeaturesTransformer()
    result = transformer.fit_transform(X_train.iloc[:5])
    
    assert result.shape[0] == 5
    assert result.shape[1] > 0


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_with_features():
    """Test TSFeaturesTransformer with specific features."""
    X_train, _ = load_basic_motions(split="train")
    
    transformer = TSFeaturesTransformer(features=["hurst", "stability"])
    result = transformer.fit_transform(X_train.iloc[:3])
    
    assert result.shape[0] == 3
    assert result.shape[1] >= 2

