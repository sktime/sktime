# -*- coding: utf-8 -*-
"""Tests for time series k-means."""
import numpy as np
from sklearn import metrics

from sktime.clustering._k_means import TimeSeriesKMeans
from sktime.datasets import load_basic_motions

expected_results = {
    "mean": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        5,
        1,
        1,
        1,
        1,
        1,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
}

expected_score = {"mean": 0.2858974358974359}

expected_iters = {"mean": 300}

expected_labels = {
    "mean": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        4,
        7,
        5,
        2,
        1,
        5,
        3,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        6,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        1,
        1,
    ]
}


def test_kmeans():
    """Test implementation of Kmeans."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kmeans = TimeSeriesKMeans(averaging_method="mean", random_state=1)
    kmeans.fit(X_train)
    test_mean_result = kmeans.predict(X_test)
    mean_score = metrics.rand_score(y_test, test_mean_result)
    proba = kmeans.predict_proba(X_test)

    assert np.array_equal(test_mean_result, expected_results["mean"])
    assert mean_score == expected_score["mean"]
    assert kmeans.n_iter_ == 300
    assert np.array_equal(kmeans.labels_, expected_labels["mean"])
    assert isinstance(kmeans.cluster_centers_, np.ndarray)
    assert proba.shape == (40, 6)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
