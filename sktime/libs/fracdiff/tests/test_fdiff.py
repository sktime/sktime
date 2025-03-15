import math

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

from sktime.libs.fracdiff import fdiff, fdiff_coef
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.fracdiff"),
    reason="Execute tests for fracdiff iff anything in the module has changed",
)
class TestFdiff:
    """
    Test `fdiff`.
    """

    @staticmethod
    def _pochhammer(d, n):
        """
        Pochhammer symbol

        >>> TestFdiff._pochhammer(5, 3)  # 5 * 4 * 3
        60
        """
        return np.prod([d - k for k in range(n)])

    @staticmethod
    def _coef(n, window):
        """
        Different yet more direct way to compute coefficients.

        >>> TestFdiff._coef(0.5, 4)
        array([ 1.    , -0.5   , -0.125 , -0.0625])
        """
        i = np.arange(window)
        s = (-1) ** i
        p = np.array([TestFdiff._pochhammer(n, ii) for ii in i])
        f = np.array([math.factorial(ii) for ii in i])
        return s * p / f

    @pytest.mark.parametrize("n", [0.0, 0.1, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("window", [1, 2, 10])
    def test_coef(self, n, window):
        out = self._coef(n, window)
        assert_allclose(fdiff_coef(n, window), out)

    def test_error_0dim(self):
        assert_raises(ValueError, fdiff, np.array(0), n=0.5)

    def test_error_invalid_mode(self):
        assert_raises(ValueError, fdiff, np.zeros(100), n=0.5, mode="invalid")

    @pytest.mark.parametrize("n", [0, 1, 2, 1.0])  # TODO 1.0 ...
    # @pytest.mark.parametrize("window", [1,2,3])  # TODO
    @pytest.mark.parametrize("ndim", [1, 2, 3])  # TODO 1.0 ...
    def test_integer_n(self, n, ndim):
        """
        fdiff with integer n is equal to numpy.diff
        """
        np.random.seed(42)
        a = np.random.randn(*([100] * ndim))
        for axis in range(ndim):
            np_diff_res = np.diff(a, n=int(n), axis=axis)

            slices = [slice(None)] * a.ndim
            slices[axis] = slice(int(n), None)
            slices = tuple(slices)

            last_fdiff_res = fdiff(a, n=n, axis=axis)[slices]

            assert_allclose(np_diff_res, last_fdiff_res)

    @pytest.mark.parametrize("n", [0.5])
    @pytest.mark.parametrize("window", [2])
    @pytest.mark.parametrize("n_blanks_1", [0, 1])
    @pytest.mark.parametrize("n_blanks_2", [0, 1])
    @pytest.mark.parametrize("n_features", [1])
    def test_one(self, n, window, n_blanks_1, n_blanks_2, n_features):
        a = np.concatenate(
            (
                np.zeros((window + n_blanks_1, n_features)),
                np.ones((1, n_features)),
                np.zeros((window + n_blanks_2, n_features)),
            )
        )

        coef = fdiff_coef(n, window)
        diff = fdiff(a, n=n, window=window, axis=0)

        for i in range(n_features):
            assert_array_equal(diff[window + n_blanks_1 :, i][:window], coef)
            assert_equal(diff[: window + n_blanks_1], 0)
            assert_equal(diff[window + n_blanks_1 + window :], 0)

    @pytest.mark.parametrize("n", [0.5, 1.5])
    def test_axis(self, n):
        np.random.seed(42)
        a = np.random.randn(10, 20, 30, 40)

        out1 = np.swapaxes(fdiff(np.swapaxes(a, 0, 1), n, axis=0), 0, 1)
        out2 = np.swapaxes(fdiff(np.swapaxes(a, 0, 2), n, axis=0), 0, 2)
        out3 = np.swapaxes(fdiff(np.swapaxes(a, 0, 3), n, axis=0), 0, 3)

        assert_array_equal(fdiff(a, n, axis=1), out1)
        assert_array_equal(fdiff(a, n, axis=2), out2)
        assert_array_equal(fdiff(a, n, axis=3), out3)
        assert_array_equal(fdiff(a, n, axis=0), fdiff(a, n, axis=-4))
        assert_array_equal(fdiff(a, n, axis=1), fdiff(a, n, axis=-3))
        assert_array_equal(fdiff(a, n, axis=2), fdiff(a, n, axis=-2))
        assert_array_equal(fdiff(a, n, axis=3), fdiff(a, n, axis=-1))

    @pytest.mark.parametrize("n", [0.5, 1.5])
    @pytest.mark.parametrize("window", [2])
    def test_mode(self, n, window):
        np.random.seed(42)
        a = np.random.randn(10, 20)

        out_s = fdiff(a, n, axis=0, window=window, mode="same")
        out_v = fdiff(a, n, axis=0, window=window, mode="valid")
        assert_array_equal(out_s[window - 1 :, :], out_v)

        out_s = fdiff(a, n, axis=1, window=window, mode="same")
        out_v = fdiff(a, n, axis=1, window=window, mode="valid")
        assert_array_equal(out_s[:, window - 1 :], out_v)

    @pytest.mark.parametrize("n", [0.5, 1.5])
    @pytest.mark.parametrize("window", [2])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    def test_prepend(self, n, window, mode):
        a = np.random.randn(10, 20)

        prepend = np.random.randn(5, 20)
        cat = np.concatenate((prepend, a), axis=0)
        out = fdiff(cat, n, axis=0, window=window, mode=mode)
        result = fdiff(a, n, axis=0, window=window, prepend=prepend, mode=mode)
        assert_array_equal(result, out)

        prepend = np.random.randn(10, 5)
        cat = np.concatenate((prepend, a), axis=1)
        out = fdiff(cat, n, axis=1, window=window, mode=mode)
        result = fdiff(a, n, axis=1, window=window, prepend=prepend, mode=mode)
        assert_array_equal(result, out)

        cat = np.concatenate((np.zeros((1, 20)), a), axis=0)
        out = fdiff(cat, n, axis=0, window=window, mode=mode)
        result = fdiff(a, n, axis=0, window=window, prepend=0, mode=mode)
        assert_array_equal(result, out)

        cat = np.concatenate((np.zeros((10, 1)), a), axis=1)
        out = fdiff(cat, n, axis=1, window=window, mode=mode)
        result = fdiff(a, n, axis=1, window=window, prepend=0, mode=mode)
        assert_array_equal(result, out)

    @pytest.mark.parametrize("n", [0.5, 1.5])
    @pytest.mark.parametrize("window", [2])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    def test_append(self, n, window, mode):
        a = np.random.randn(10, 20)

        append = np.random.randn(5, 20)
        cat = np.concatenate((a, append), axis=0)
        out = fdiff(cat, n, axis=0, window=window, mode=mode)
        result = fdiff(a, n, axis=0, window=window, append=append, mode=mode)
        assert_array_equal(result, out)

        append = np.random.randn(10, 5)
        cat = np.concatenate((a, append), axis=1)
        out = fdiff(cat, n, axis=1, window=window, mode=mode)
        result = fdiff(a, n, axis=1, window=window, append=append, mode=mode)
        assert_array_equal(result, out)

        cat = np.concatenate((a, np.zeros((1, 20))), axis=0)
        out = fdiff(cat, n, axis=0, window=window, mode=mode)
        result = fdiff(a, n, axis=0, window=window, append=0, mode=mode)
        assert_array_equal(result, out)

        cat = np.concatenate((a, np.zeros((10, 1))), axis=1)
        out = fdiff(cat, n, axis=1, window=window, mode=mode)
        result = fdiff(a, n, axis=1, window=window, append=0, mode=mode)
        assert_array_equal(result, out)

    @pytest.mark.parametrize("n", [0.5, 1.5])
    @pytest.mark.parametrize("window", [2])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    def test_linearity_add(self, n, window, mode):
        np.random.seed(42)
        a0 = np.random.randn(10, 20)
        a1 = np.random.randn(10, 20)
        a2 = a0 + a1
        diff0 = fdiff(a0, n, window=window, mode=mode)
        diff1 = fdiff(a1, n, window=window, mode=mode)
        diff2 = fdiff(a2, n, window=window, mode=mode)
        assert_allclose(diff0 + diff1, diff2)

    @pytest.mark.parametrize("n", [0.5, 1.5])
    @pytest.mark.parametrize("window", [2])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    @pytest.mark.parametrize("const", [2, 0.5, -1])
    def test_linearity_mul(self, n, window, mode, const):
        np.random.seed(42)
        a0 = np.random.randn(10, 20)
        a1 = const * a0
        diff0 = fdiff(a0, n, window=window, mode=mode)
        diff1 = fdiff(a1, n, window=window, mode=mode)
        assert_allclose(const * diff0, diff1)

    def test_full_deprecated(self):
        np.random.seed(42)
        X = np.random.randn(10, 10)
        with pytest.raises(DeprecationWarning):
            _ = fdiff(X, 0.5, mode="full")
