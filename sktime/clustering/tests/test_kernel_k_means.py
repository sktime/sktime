"""Tests for time series kernel kmeans."""
import numpy as np
import pytest

from sktime.clustering.kernel_k_means import TimeSeriesKernelKMeans
from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class

expected_results = [
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

expected_score = 73.99999999999983

expected_iters = 2

expected_labels = [
    1,
    0,
    0,
    1,
    2,
    1,
    0,
    2,
    2,
    1,
    1,
    1,
    0,
    0,
    1,
    0,
    2,
    0,
    0,
    1,
    1,
    2,
    0,
    0,
    1,
    2,
    2,
    1,
    0,
    2,
    1,
    2,
    1,
    0,
    1,
    1,
    2,
    0,
    0,
    2,
]


@pytest.mark.skipif(
    not run_test_for_class(TimeSeriesKernelKMeans),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_kernel_k_means():
    """Test implementation of kernel k means."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kernel_kmeans = TimeSeriesKernelKMeans(random_state=1, n_clusters=3)
    kernel_kmeans.fit(X_train)
    test_shape_result = kernel_kmeans.predict(X_test)
    score = kernel_kmeans.score(X_test)
    proba = kernel_kmeans.predict_proba(X_test)

    assert np.array_equal(test_shape_result, expected_results)
    np.testing.assert_almost_equal(score, expected_score)
    assert kernel_kmeans.n_iter_ == expected_iters
    assert np.array_equal(kernel_kmeans.labels_, expected_labels)
    assert proba.shape == (40, 3)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
