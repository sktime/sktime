"""Tests for ClaSPTransformer."""

__author__ = ["NikunjSharma-dev"]

import numpy as np
from sktime.transformations.clasp import ClaSPTransformer


def test_clasp_transformer_default_params():
    """Verify that ClaSPTransformer works with default parameters under NumPy 1.x & 2.x."""
    X = np.arange(100)
    transformer = ClaSPTransformer()
    res = transformer.fit_transform(X)
    assert res is not None
    assert len(res) > 0


if __name__ == "__main__":
    test_clasp_transformer_default_params()
    print("test_clasp_transformer_default_params PASSED!")
