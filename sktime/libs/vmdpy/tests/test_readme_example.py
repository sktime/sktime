"""Tests for vmdpy package - readme example."""

import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.vmdpy"),
    reason="skip test if module not changed",
)
def test_readme_example():
    """Tests the example from the README.md"""
    import numpy as np

    from sktime.libs.vmdpy import VMD

    # Time Domain 0 to T
    T = 1000
    fs = 1 / T
    t = np.arange(1, T + 1) / T
    freqs = 2 * np.pi * (t - 0.5 - fs) / (fs)  # noqa: F841

    # center frequencies of components
    f_1 = 2
    f_2 = 24
    f_3 = 288

    # modes
    v_1 = np.cos(2 * np.pi * f_1 * t)
    v_2 = 1 / 4 * (np.cos(2 * np.pi * f_2 * t))
    v_3 = 1 / 16 * (np.cos(2 * np.pi * f_3 * t))

    f = v_1 + v_2 + v_3 + 0.1 * np.random.randn(v_1.size)

    # some sample parameters for VMD
    alpha = 2000  # moderate bandwidth constraint
    tau = 0.0  # noise-tolerance (no strict fidelity enforcement)
    K = 3  # 3 modes
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7

    # Run VMD
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)

    if not _check_soft_dependencies("matplotlib", severity="none"):
        return None

    import matplotlib.pyplot as plt

    # Visualize decomposed modes
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f)
    plt.title("Original signal")
    plt.xlabel("time (s)")
    plt.subplot(2, 1, 2)
    plt.plot(u.T)
    plt.title("Decomposed modes")
    plt.xlabel("time (s)")
    plt.legend(["Mode %d" % m_i for m_i in range(u.shape[0])])
    plt.tight_layout()
