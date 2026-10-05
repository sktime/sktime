"""Regression tests for the memory-efficient VMD implementation."""

import numpy as np
import pytest

from sktime.libs.vmdpy import VMD


@pytest.mark.parametrize("length", [1000, 1001])
@pytest.mark.parametrize("dc", [False, True])
def test_vmd_shapes_and_finite_values(length, dc):
    """VMD preserves the reference output contract for both signal parities."""
    t = np.linspace(0, 1, length, endpoint=False)
    signal = np.sin(2 * np.pi * 3 * t) + 0.25 * np.cos(2 * np.pi * 11 * t)

    modes, modes_hat, omega = VMD(
        signal,
        alpha=2000,
        tau=0,
        K=3,
        DC=dc,
        init=1,
        tol=1e-7,
    )

    assert modes.shape == (3, length)
    assert modes_hat.shape == (length, 3)
    assert omega.shape[1] == 3
    assert np.isfinite(modes).all()
    assert np.isfinite(modes_hat).all()
    assert np.isfinite(omega).all()
    if dc:
        np.testing.assert_allclose(omega[:, 0], 0)
