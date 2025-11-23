"""Tests for TSFeaturesTransformer."""

__author__ = ["Faakhir30"]

import numpy as np
import pytest

from sktime.datasets import load_arrow_head
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.panel.tsfeatures import TSFeaturesTransformer
from sktime.utils._testing.panel import make_classification_problem


@pytest.mark.skipif(
    not run_test_for_class(TSFeaturesTransformer),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_tsfeatures_extractor():
    """Test that TSFeaturesTransformer extracts features correctly."""
    X, _ = make_classification_problem()

    transformer = TSFeaturesTransformer(freq=1, scale=True, threads=1)

    Xt = transformer.fit_transform(X)

    # Check that output is a DataFrame
    assert isinstance(Xt, type(X))
    # Check that we have features (columns)
    assert Xt.shape[1] > 0
    # Check that we have the same number of instances (rows)
    assert Xt.shape[0] == X.shape[0]
    # Check that all values are numeric (no object dtype)
    assert Xt.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
