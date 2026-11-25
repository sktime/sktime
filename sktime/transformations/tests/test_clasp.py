"""Tests for ClaSPTransformer."""

import numpy as np

from sktime.transformations.clasp import ClaSPTransformer


def test_clasp_f1_zero_division():
    """Test that ClaSPTransformer does not raise ZeroDivisionError for tp=0."""
    a = np.arange(100)
    c = ClaSPTransformer(scoring_metric="F1")
    # This previously raised ZeroDivisionError
    c.fit_transform(a)
