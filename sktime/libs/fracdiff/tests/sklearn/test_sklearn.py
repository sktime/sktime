import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sktime.libs.fracdiff.sklearn import Fracdiff
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.fracdiff.sklearn"),
    reason="Execute tests for fracdiff.sklearn iff anything in the module has changed",
)
class TestScikitLearn:
    @pytest.mark.parametrize("seed", [42])
    @pytest.mark.parametrize("n_samples", [20, 100])
    @pytest.mark.parametrize("n_features", [1, 10])
    @pytest.mark.parametrize("d", [0.5])
    def test_sample_fit_transform(self, seed, n_samples, n_features, d):
        np.random.seed(seed)

        X = np.random.randn(n_samples, n_features)
        _ = Fracdiff(d).fit_transform(X)

    @pytest.mark.parametrize("seed", [42])
    @pytest.mark.parametrize("n_samples", [20, 100])
    @pytest.mark.parametrize("n_features", [1, 10])
    @pytest.mark.parametrize("d", [0.5])
    def test_sample_pipeline(self, seed, n_samples, n_features, d):
        np.random.seed(seed)

        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("fracdiff", Fracdiff(d)),
                ("regressor", LinearRegression()),
            ]
        )

        pipeline.fit(X, y)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.fracdiff.sklearn"),
    reason="Execute tests for fracdiff.sklearn iff anything in the module has changed",
)
def test_window_from_tol_coef():
    """Test the window_from_tol_coef function."""
    from numpy.testing import assert_equal

    from sktime.libs.fracdiff import fdiff_coef
    from sktime.libs.fracdiff.sklearn.tol import window_from_tol_coef

    assert_equal(window_from_tol_coef(0.5, 0.1), 4)
    assert_equal(fdiff_coef(0.5, 3)[-1], -0.125)
    assert_equal(fdiff_coef(0.5, 4)[-1], -0.0625)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs.fracdiff.sklearn"),
    reason="Execute tests for fracdiff.sklearn iff anything in the module has changed",
)
def test_window_from_tol_memory():
    """Test the window_from_tol_memory function."""
    from numpy.testing import assert_almost_equal

    from sktime.libs.fracdiff import fdiff_coef
    from sktime.libs.fracdiff.sklearn.tol import window_from_tol_memory

    assert window_from_tol_memory(0.5, 0.2) == 9
    assert_almost_equal(np.sum(fdiff_coef(0.5, 10000)[9:]), -0.19073, decimal=5)
    assert_almost_equal(np.sum(fdiff_coef(0.5, 10000)[8:]), -0.20383, decimal=5)
