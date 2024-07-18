import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sktime.tests.test_switch import run_test_module_changed

from sktime.libs.fracdiff import fdiff
from sktime.libs.fracdiff.sklearn import Fracdiff


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.fracdiff.sklearn"),
    reason="Execute tests for fracdiff.sklearn iff anything in the module has changed",
)
class TestFracdiff:
    def test_repr(self):
        fracdiff = Fracdiff(0.5, window=10, mode="same", window_policy="fixed")
        expected = "Fracdiff(d=0.5, window=10, mode=same, window_policy=fixed)"
        assert repr(fracdiff) == expected

    @pytest.mark.parametrize("d", [0.5])
    @pytest.mark.parametrize("window", [10])
    @pytest.mark.parametrize("mode", ["same", "valid"])
    def test_transform(self, d, window, mode):
        np.random.seed(42)
        X = np.random.randn(50, 100)
        fracdiff = Fracdiff(d=d, window=window, mode=mode)
        out = fdiff(X, n=d, axis=0, window=window, mode=mode)
        assert_array_equal(fracdiff.fit_transform(X), out)
