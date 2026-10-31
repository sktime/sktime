"""Tests for the low-rank variant of the signature kernel."""

import numpy as np
import pytest

from sktime.dists_kernels.signature_kernel import (
    SignatureKernel,
    sqize_kernel_low_rank_fast,
)
from sktime.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(SignatureKernel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("level", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "normalize, theta", [(False, 1.0), (False, 2.0), (True, 1.0), (True, 0.3)]
)
def test_lowrank_full_rank_equals_exact(level, normalize, theta):
    """Without rank reduction, the low-rank kernel equals the exact kernel.

    Checked for the kernel matrix of X, and between X and X2 of different length.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 2, 6))
    X2 = rng.standard_normal((3, 2, 9))

    params = {"level": level, "normalize": normalize, "theta": theta}
    exact = SignatureKernel(**params)
    lowrank = SignatureKernel(lowrank=True, **params)

    K = exact.transform(X)
    np.testing.assert_allclose(lowrank.transform(X), K, rtol=1e-10, atol=1e-12)

    K_XX2 = exact.transform(X, X2)
    np.testing.assert_allclose(lowrank.transform(X, X2), K_XX2, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(
    not run_test_for_class(SignatureKernel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lowrank_level_one():
    """Level 1 low-rank features are 1 and the sum of the points of the series."""
    rng = np.random.default_rng(1)
    K = rng.standard_normal((3, 5, 2))  # 3 series, 5 time points, 2 features

    R = sqize_kernel_low_rank_fast(K, L=1)

    np.testing.assert_allclose(R[:, 0], 1.0)
    np.testing.assert_allclose(R[:, 1:], K.sum(axis=1))


@pytest.mark.skipif(
    not run_test_for_class(SignatureKernel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lowrank_rankbound():
    """With a rankbound, each level has at most rankbound features."""
    rng = np.random.default_rng(2)
    K = rng.standard_normal((3, 5, 2))

    R = sqize_kernel_low_rank_fast(K, L=3, rankbound=3)

    # 1 for level 0, 2 for level 1, then at most 3 for levels 2 and 3
    assert R.shape == (3, 1 + 2 + 3 + 3)
