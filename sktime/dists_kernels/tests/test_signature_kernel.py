"""Tests for the signature kernel, in particular its higher order variant."""

import itertools
import math

import numpy as np
import pytest

from sktime.dists_kernels.signature_kernel import (
    SignatureKernel,
    sqize_kernel,
    sqize_kernel_ho,
)
from sktime.tests.test_switch import run_test_for_class


def _index_sequences(n, level, degree):
    """Non-decreasing index sequences with their higher order weights.

    Yields all non-decreasing tuples of length ``level`` with entries in
    ``range(n)``, where no index is repeated more than ``degree`` times,
    together with the weight ``1 / prod(multiplicity!)`` of the order ``degree``
    approximation of the signature, Definition 7.5 in Kiraly and Oberhauser.
    """
    for seq in itertools.combinations_with_replacement(range(n), level):
        multiplicities = [seq.count(i) for i in set(seq)]
        if max(multiplicities) <= degree:
            yield seq, 1 / math.prod(math.factorial(m) for m in multiplicities)


def _sqize_kernel_ho_brute_force(K, L, D):
    """Higher order sequential kernel by summing over all index sequences."""
    n_rows, n_cols = K.shape
    result = 1.0
    for level in range(1, L + 1):
        for seq_i, w_i in _index_sequences(n_rows, level, D):
            for seq_j, w_j in _index_sequences(n_cols, level, D):
                result += w_i * w_j * math.prod(K[i, j] for i, j in zip(seq_i, seq_j))
    return result


@pytest.mark.skipif(
    not run_test_for_class(SignatureKernel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("L", [1, 2, 3, 4])
@pytest.mark.parametrize("D", [1, 2, 3])
@pytest.mark.parametrize("shape", [(3, 4), (5, 5)])
def test_sqize_kernel_ho_brute_force(L, D, shape):
    """The higher order sequential kernel equals its definition."""
    rng = np.random.default_rng(42)
    K = rng.standard_normal(shape)

    expected = _sqize_kernel_ho_brute_force(K, L, D)
    np.testing.assert_allclose(sqize_kernel_ho(K, L, D), expected, rtol=1e-10)


@pytest.mark.skipif(
    not run_test_for_class(SignatureKernel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("L", [1, 2, 3, 4])
@pytest.mark.parametrize("normalize, theta", [(False, 1.0), (True, 1.0), (True, 0.5)])
def test_sqize_kernel_ho_order_one(L, normalize, theta):
    """For order D=1, the higher order kernel is the sequential kernel.

    See Remark 7.6 in Kiraly and Oberhauser, "Kernels for sequentially ordered data".
    """
    rng = np.random.default_rng(0)
    K = rng.standard_normal((6, 4))

    kwargs = {"theta": theta, "normalize": normalize}
    np.testing.assert_allclose(
        sqize_kernel_ho(K, L, D=1, **kwargs), sqize_kernel(K, L, **kwargs), rtol=1e-12
    )


@pytest.mark.skipif(
    not run_test_for_class(SignatureKernel),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("level", [2, 3, 4])
@pytest.mark.parametrize("normalize", [False, True])
def test_signature_kernel_higher_order(level, normalize):
    """SignatureKernel with degree > 1 runs, is symmetric and positive semi-definite.

    Also checks that the kernel matrix between X and X2 is consistent.
    """
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 2, 8))

    kernel = SignatureKernel(level=level, degree=2, normalize=normalize)
    K = kernel.transform(X)

    assert K.shape == (5, 5)
    np.testing.assert_allclose(K, K.T, rtol=1e-10)
    assert np.linalg.eigvalsh(K).min() > -1e-8 * np.abs(K).max()

    K_XY = kernel.transform(X, X[:3])
    np.testing.assert_allclose(K_XY, K[:, :3], rtol=1e-10)
