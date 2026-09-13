"""Tests for DistFromAligner symmetry."""

import numpy as np

from sktime.alignment.lucky import AlignerLuckyDtw
from sktime.datasets import load_unit_test
from sktime.dists_kernels.compose_from_align import DistFromAligner


def test_dist_from_aligner_symmetric_when_x2_none():
    """DistFromAligner must return a symmetric matrix when X2 is not passed.

    Regression test for #11091: the symmetry shortcut in
    DistFromAligner._transform was never triggered because the base class
    transform() always passes X2 explicitly. With an aligner whose distance
    depends on argument order (AlignerLuckyDtw tie-breaking), the resulting
    self-distance matrix was asymmetric despite the ``symmetric: True`` tag.
    """
    X, _ = load_unit_test()
    X = X[0:3]

    dist_mat = DistFromAligner(AlignerLuckyDtw()).transform(X)

    assert dist_mat.shape == (3, 3)
    assert np.allclose(dist_mat, dist_mat.T), (
        "self-distance matrix should be symmetric, got:\n"
        f"{dist_mat}"
    )


def test_dist_from_aligner_full_matrix_when_x2_passed():
    """DistFromAligner must compute the full matrix when a distinct X2 is passed."""
    X, _ = load_unit_test()
    X = X[0:3]
    X2 = X[0:2]

    dist_mat = DistFromAligner(AlignerLuckyDtw()).transform(X, X2)

    assert dist_mat.shape == (3, 2)
