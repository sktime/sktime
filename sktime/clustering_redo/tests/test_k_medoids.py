# -*- coding: utf-8 -*-
"""Tests for time series k-medoids."""
import numpy as np
from sklearn import metrics

from sktime.clustering_redo._k_medoids import TimeSeriesKMedoids
from sktime.datasets import load_UCR_UEA_dataset

dataset_name = "Beef"

expected_results = {
    "medoids": [
        4,
        7,
        7,
        7,
        7,
        6,
        4,
        2,
        1,
        5,
        2,
        2,
        4,
        2,
        4,
        4,
        2,
        2,
        1,
        3,
        1,
        4,
        3,
        3,
        4,
        0,
        6,
        1,
        2,
        0,
    ]
}

expected_score = {"medoids": 0.7839080459770115}

expected_iters = {"medoids": 300}

expected_labels = {
    "medoids": [
        6,
        7,
        7,
        7,
        7,
        6,
        4,
        3,
        1,
        6,
        2,
        2,
        4,
        2,
        4,
        4,
        0,
        2,
        1,
        3,
        5,
        4,
        3,
        3,
        4,
        0,
        6,
        1,
        2,
        0,
    ]
}


def test_kmedoids():
    """Test implementation of Kmeans."""
    X_train, y_train = load_UCR_UEA_dataset(
        dataset_name, split="train", return_X_y=True
    )
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    kmedoids = TimeSeriesKMedoids(random_state=1)
    kmedoids.fit(X_train)
    test_mean_result = kmedoids.predict(X_test)
    medoids_score = metrics.rand_score(y_test, test_mean_result)

    assert np.array_equal(test_mean_result, expected_results["medoids"])
    assert medoids_score == expected_score["medoids"]
    assert kmedoids.n_iter_ == 300
    assert np.array_equal(kmedoids.labels_, expected_labels["medoids"])
    assert isinstance(kmedoids.cluster_centers_, np.ndarray)
