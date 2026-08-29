"""Groundtruth tests for MiniRocketMultivariateCython vs the numba reference."""

import numpy as np
import pytest

from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.rocket import (
    MiniRocketMultivariate,
    MiniRocketMultivariateCython,
)


@pytest.mark.skipif(
    not run_test_for_class(MiniRocketMultivariateCython),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("n_columns", [1, 4])
@pytest.mark.parametrize(
    "num_kernels,max_dilations_per_kernel,random_state",
    [(84, 32, 42), (168, 16, 7), (84, 32, 0)],
)
def test_cython_matches_numba(
    n_columns, num_kernels, max_dilations_per_kernel, random_state
):
    """Cython transform must match the numba implementation (groundtruth)."""
    rng = np.random.RandomState(random_state)
    X = rng.normal(size=(6, n_columns, 60)).astype(np.float32)

    kw = dict(
        num_kernels=num_kernels,
        max_dilations_per_kernel=max_dilations_per_kernel,
        random_state=random_state,
    )
    numba_out = MiniRocketMultivariate(**kw).fit_transform(X)
    cython_out = MiniRocketMultivariateCython(**kw).fit_transform(X)

    assert cython_out.shape == numba_out.shape
    np.testing.assert_allclose(
        cython_out.to_numpy(), numba_out.to_numpy(), rtol=1e-4, atol=1e-5
    )
