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
