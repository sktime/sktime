# -*- coding: utf-8 -*-
"""Tests for time series kernel k-means."""
import numpy as np
from sklearn import metrics

from sktime.clustering._kernel_k_means import KernelKMeans
from sktime.datasets import load_basic_motions

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

inertia = 73.99999999999983

expected_score = 0.28717948717948716

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


def test_kernel_k_means():
    """Test implementation of Kshapes."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kernel_kmeans = KernelKMeans(random_state=1, n_clusters=3, verbose=True)
    kernel_kmeans.fit(X_train)
    test_shape_result = kernel_kmeans.predict(X_test)
    score = metrics.rand_score(y_test, test_shape_result)
    proba = kernel_kmeans.predict_proba(X_test)

    assert np.array_equal(test_shape_result, expected_results)
    assert score == expected_score
    assert kernel_kmeans.n_iter_ == expected_iters
    assert np.array_equal(kernel_kmeans.labels_, expected_labels)
    assert proba.shape == (40, 2)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
