# -*- coding: utf-8 -*-
"""Tests for time series k-medoids."""
import numpy as np
from sklearn import metrics

from sktime.clustering._k_medoids import TimeSeriesKMedoids
from sktime.datasets import load_basic_motions

expected_results = {
    "medoids": [
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
        3,
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

expected_score = {"medoids": 0.2858974358974359}

expected_iters = {"medoids": 300}

expected_labels = {
    "medoids": [
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


def test_kmedoids():
    """Test implementation of Kmedoids."""
    X_train, y_train = load_basic_motions(split="train")
    X_test, y_test = load_basic_motions(split="test")

    kmedoids = TimeSeriesKMedoids(random_state=1)
    kmedoids.fit(X_train)
    test_medoids_result = kmedoids.predict(X_test)
    medoids_score = metrics.rand_score(y_test, test_medoids_result)
    proba = kmedoids.predict_proba(X_test)

    assert np.array_equal(test_medoids_result, expected_results["medoids"])
    assert medoids_score == expected_score["medoids"]
    assert kmedoids.n_iter_ == 300
    assert np.array_equal(kmedoids.labels_, expected_labels["medoids"])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
    assert proba.shape == (40, 5)

    for val in proba:
        assert np.count_nonzero(val == 1.0) == 1
