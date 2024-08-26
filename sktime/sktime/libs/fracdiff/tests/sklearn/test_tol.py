import numpy as np
import pytest

from sktime.libs.fracdiff.fdiff import fdiff_coef
from sktime.libs.fracdiff.sklearn.tol import (
    window_from_tol_coef,
    window_from_tol_memory,
)
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.fracdiff.sklearn"),
    reason="Execute tests for fracdiff.sklearn iff anything in the module has changed",
)
class TestTol:
    LARGE = 10**6

    @pytest.mark.parametrize("d", [0.0, 0.1, 0.5, 1.0, 1.5])
    @pytest.mark.parametrize("tol", [0.1, 0.01])
    def test_tol_coef(self, d, tol):
        window = window_from_tol_coef(d, tol)

        assert np.abs(fdiff_coef(d, window - 1)[-1]) > tol
        assert np.abs(fdiff_coef(d, window)[-1]) < tol

    @pytest.mark.parametrize("d", [0.5])
    @pytest.mark.parametrize("tol", [0.1])
    def test_tol_memory(self, d, tol):
        window = window_from_tol_memory(d, tol)

        lost_memory_0 = np.abs(np.sum(fdiff_coef(d, self.LARGE)[window:]))
        lost_memory_1 = np.abs(np.sum(fdiff_coef(d, self.LARGE)[window - 1 :]))

        assert lost_memory_0 < tol
        assert lost_memory_1 > tol
